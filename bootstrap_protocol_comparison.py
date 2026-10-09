"""
bootstrap_protocol_comparison.py

Bootstrap analysis of regression coefficients across the corrected
protocols (nested, shared-skel, non-nested) to determine whether the
apparent coefficient differences are statistically robust or within
sampling noise.

Background
----------
The targeted sweeps produced point estimates:

    Protocol     alpha     beta_F   beta_N    SE(beta_N)
    nested      -22.93     +33.20    -0.210     0.044
    shared_skel -18.75     +25.71    -0.288     0.048
    non_nested  -22.90     +30.39    -0.210     0.054

The shared-skel vs non-nested difference on beta_N is 0.078, only
about 1.1 joint SE.  Before committing 17 hours to a replication,
this bootstrap settles whether the gap is robust within the existing
50 seeds per cell.

Method
------
For each protocol's 450-row sweep CSV (9 cells x 50 seeds):

  1. WITHIN-CELL BOOTSTRAP (primary):  Keep the 9 cells fixed.
     Within each cell, resample the 50 seeds with replacement.
     Refit P(FI) ~ logit(alpha + beta_F * F_star + beta_N * N).
     Record coefficients.  Repeat N_BOOT times.

  2. ROW-LEVEL BOOTSTRAP (sanity check):  Resample all 450 rows
     with replacement, refit, record.  Reported as a consistency
     check on the within-cell results.

For each protocol pair (shared-skel vs non-nested, etc.):

  3. Compute the bootstrap distribution of the coefficient
     difference (beta^A - beta^B) by pairing the i-th bootstrap
     of A with the i-th bootstrap of B.  This gives a paired-
     differences distribution.

  4. Report 90%, 95%, 99% percentile CIs on each protocol's
     coefficients AND on all pairwise differences.

  5. Plot bootstrap distributions side by side for visual
     inspection.

Usage
-----
    python3 bootstrap_protocol_comparison.py

The script reads the two CSVs from /mnt/user-data/uploads/ by
default.  Override via the --shared_skel and --non_nested flags.
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import statsmodels.api as sm
import matplotlib.pyplot as plt


# ==============================================================
# Defaults
# ==============================================================
DEFAULT_SHARED_SKEL_CSV = (
    './logs/bootstrap_protocol_comparison/phase2_shared_skel-60x30-60x30-S=1.csv'
)

DEFAULT_SHARED_SKEL_STOCH_CSV = (
    './logs/bootstrap_protocol_comparison/phase4_shared_skel_stoch-60x30-60x30-S=1.csv'
)

DEFAULT_NON_NESTED_CSV = (
    './logs/bootstrap_protocol_comparison/phase3_non_nested-60x30-60x30-S=1.csv'
)
DEFAULT_NESTED_CSV = (
    './logs/bootstrap_protocol_comparison/targeted_nested_sweep-60x30-60x30-S=1.csv'
)
N_BOOTSTRAPS = 2000
RANDOM_SEED = 42

# Reference coefficients from the nested re-run (60x30, n=50).
# We treat these as fixed reference values because we don't have the
# raw nested CSV; users with the nested CSV can pass it via --nested.
NESTED_REFERENCE = {
    'alpha':   -22.93,
    'beta_F':  +33.20,
    'beta_N':  -0.210,
    'se_alpha': 3.51,
    'se_beta_F': 5.02,
    'se_beta_N': 0.044,
}

# Flat-schedule (fixed-rho i.i.d.) reference coefficients (60x30 dense
# sweep).  Used to locate the stochastic-severity protocol's beta_N
# between the graduated class (~ -0.21 to -0.29) and the flat value
# (H3 of the Part II registered program).
FLAT_REFERENCE = {
    'alpha':   -14.07,
    'beta_F':  +19.84,
    'beta_N':  -0.580,
    'se_beta_N': 0.079,
}


# ==============================================================
# Data loading and outcome labeling
# ==============================================================

def load_protocol_csv(path, label):
    """Load a sweep CSV and add the fi_holds outcome column."""
    df = pd.read_csv(path)
    # The targeted-sweep analyzer labels FI as 'marginal' or 'free'
    df['fi_holds'] = df['C_interp'].isin(['marginal', 'free']).astype(int)
    print(f"  Loaded {label}: {len(df)} rows, "
          f"{df.groupby(['rho', 'N']).ngroups} unique cells")
    n_pos = df['fi_holds'].sum()
    print(f"    P(FI) overall: {n_pos / len(df):.4f}  "
          f"({n_pos} positive, {len(df) - n_pos} negative)")
    return df


# ==============================================================
# Logistic regression
# ==============================================================

def fit_logistic_regression(df):
    """Fit P(FI) = logit(alpha + beta_F * F_star + beta_N * N).

    Returns dict {alpha, beta_F, beta_N} or None on fit failure.
    """
    df = df.dropna(subset=['F_star_density', 'N', 'fi_holds'])
    if len(df) < 4:
        return None
    n_pos = int(df['fi_holds'].sum())
    n_neg = len(df) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    try:
        X = df[['F_star_density', 'N']].values
        X = sm.add_constant(X, has_constant='add')
        y = df['fi_holds'].values
        with np.errstate(all='ignore'):
            result = sm.Logit(y, X).fit(disp=0, maxiter=50)
        if not result.mle_retvals.get('converged', True):
            return None
        return {
            'alpha':  float(result.params[0]),
            'beta_F': float(result.params[1]),
            'beta_N': float(result.params[2]),
        }
    except Exception:
        return None


# ==============================================================
# Bootstrap routines
# ==============================================================

def bootstrap_within_cell(df, n_boot, rng, label):
    """Resample seeds within each cell, refit, record coefficients."""
    cell_groups = list(df.groupby(['rho', 'N']))
    rows = []
    n_failed = 0
    for i in tqdm(range(n_boot), desc=f'within-cell {label}'):
        parts = []
        for (rho, N), cell_df in cell_groups:
            n_in_cell = len(cell_df)
            sample_idx = rng.integers(0, n_in_cell, size=n_in_cell)
            parts.append(cell_df.iloc[sample_idx])
        sample_df = pd.concat(parts, ignore_index=True)
        fit = fit_logistic_regression(sample_df)
        if fit is None:
            n_failed += 1
            continue
        fit['boot_id'] = i
        rows.append(fit)
    if n_failed > 0:
        print(f"    {n_failed}/{n_boot} fits failed in within-cell bootstrap")
    return pd.DataFrame(rows)


def bootstrap_rows(df, n_boot, rng, label):
    """Row-level bootstrap (resample all rows with replacement)."""
    n = len(df)
    rows = []
    n_failed = 0
    for i in tqdm(range(n_boot), desc=f'row-level {label}'):
        sample_idx = rng.integers(0, n, size=n)
        sample_df = df.iloc[sample_idx]
        fit = fit_logistic_regression(sample_df)
        if fit is None:
            n_failed += 1
            continue
        fit['boot_id'] = i
        rows.append(fit)
    if n_failed > 0:
        print(f"    {n_failed}/{n_boot} fits failed in row-level bootstrap")
    return pd.DataFrame(rows)


# ==============================================================
# Summary helpers
# ==============================================================

def percentile_ci(values, conf):
    """Symmetric percentile CI."""
    lo_q = (1 - conf) / 2 * 100
    hi_q = (1 + conf) / 2 * 100
    return float(np.percentile(values, lo_q)), float(np.percentile(values, hi_q))


def summarize_coefficient(values, conf_levels=(0.90, 0.95, 0.99)):
    """Return dict with mean, median, SE, and CIs."""
    vals = np.asarray(values)
    out = {
        'mean':   float(np.mean(vals)),
        'median': float(np.median(vals)),
        'se':     float(np.std(vals, ddof=1)),
        'n':      int(len(vals)),
    }
    for conf in conf_levels:
        lo, hi = percentile_ci(vals, conf)
        out[f'ci{int(conf*100)}_lo'] = lo
        out[f'ci{int(conf*100)}_hi'] = hi
    return out


def print_protocol_summary(boot_df, label):
    """Print a tidy summary block for one protocol."""
    print(f"\n  --- {label} ({len(boot_df)} bootstrap fits) ---")
    for coef in ['alpha', 'beta_F', 'beta_N']:
        s = summarize_coefficient(boot_df[coef].values)
        print(f"    {coef:>8}: median={s['median']:+8.4f}  "
              f"bootstrap SE={s['se']:.4f}  "
              f"95% CI [{s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}]")


def compute_pairwise_differences(boot_a, boot_b, label_a, label_b):
    """Paired bootstrap differences (i-th boot of A minus i-th boot of B).

    Requires both bootstrap dataframes to have the same boot_id values.
    """
    merged = boot_a.merge(boot_b, on='boot_id', suffixes=('_A', '_B'))
    diffs = {}
    for coef in ['alpha', 'beta_F', 'beta_N']:
        d = (merged[f'{coef}_A'] - merged[f'{coef}_B']).values
        s = summarize_coefficient(d)
        # Two-sided "p-value" approximation: 2 * min(P(diff>0), P(diff<0))
        p_two_sided = 2 * min((d > 0).mean(), (d < 0).mean())
        s['p_two_sided'] = float(p_two_sided)
        s['excludes_0_95'] = bool(s['ci95_lo'] > 0 or s['ci95_hi'] < 0)
        s['excludes_0_99'] = bool(s['ci99_lo'] > 0 or s['ci99_hi'] < 0)
        diffs[coef] = s
    return diffs


def print_pairwise(label_a, label_b, diffs):
    """Print a pairwise-difference table."""
    print(f"\n  --- {label_a} minus {label_b} ---")
    for coef, s in diffs.items():
        marker_95 = '*' if s['excludes_0_95'] else ' '
        marker_99 = '**' if s['excludes_0_99'] else ''
        print(f"    {coef:>8}: median={s['median']:+8.4f}  "
              f"95% CI [{s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}] "
              f"{marker_95}{marker_99}  p~{s['p_two_sided']:.3f}")
    print("    (* = 95% CI excludes 0; ** = 99% CI excludes 0)")


# ==============================================================
# Plotting
# ==============================================================

def plot_coefficient_distributions(boot_dfs, out_dir):
    """One panel per coefficient with overlaid kernel-density-style hist."""
    coefs = ['alpha', 'beta_F', 'beta_N']
    colors = {'shared_skel':       '#1f77b4',
              'non_nested':        '#ff7f0e',
              'nested':            '#2ca02c',
              'shared_skel_stoch': '#9467bd'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, coef in zip(axes, coefs):
        for label, boot_df in boot_dfs.items():
            if boot_df is None or len(boot_df) == 0:
                continue
            vals = boot_df[coef].values
            ax.hist(vals, bins=50, alpha=0.45,
                    color=colors.get(label, 'gray'), label=label,
                    density=True)
            ax.axvline(np.median(vals), color=colors.get(label, 'gray'),
                       linestyle='--', linewidth=1)
        # Reference for nested if available
        if coef in NESTED_REFERENCE:
            ax.axvline(NESTED_REFERENCE[coef], color='red',
                       linestyle=':', linewidth=2,
                       label=f'nested ref ({NESTED_REFERENCE[coef]:+.3f})')
        ax.set_xlabel(coef)
        ax.set_ylabel('density')
        ax.set_title(f'Bootstrap distribution: {coef}')
        ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'coefficient_distributions.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def plot_pairwise_differences(diff_results, out_dir, boot_dfs):
    """Plot difference distributions with vertical line at zero."""
    diffs_to_plot = []
    pairs = [('shared_skel', 'non_nested'),
             ('shared_skel_stoch', 'shared_skel'),
             ('shared_skel_stoch', 'non_nested')]
    for a, b in pairs:
        if a in boot_dfs and b in boot_dfs \
                and boot_dfs[a] is not None and boot_dfs[b] is not None:
            merged = boot_dfs[a].merge(
                boot_dfs[b], on='boot_id', suffixes=('_A', '_B'),
            )
            diffs_to_plot.append((
                f'{a} - {b}',
                {coef: (merged[f'{coef}_A'] - merged[f'{coef}_B']).values
                 for coef in ['alpha', 'beta_F', 'beta_N']},
            ))

    if not diffs_to_plot:
        return

    fig, axes = plt.subplots(len(diffs_to_plot), 3,
                             figsize=(15, 4.5 * len(diffs_to_plot)),
                             squeeze=False)
    for row_idx, (pair_label, coef_diffs) in enumerate(diffs_to_plot):
        for col_idx, coef in enumerate(['alpha', 'beta_F', 'beta_N']):
            ax = axes[row_idx][col_idx]
            d = coef_diffs[coef]
            ax.hist(d, bins=50, color='gray', alpha=0.6, density=True)
            ax.axvline(0, color='red', linestyle='-', linewidth=2,
                       label='zero (null)')
            ax.axvline(np.median(d), color='blue', linestyle='--',
                       linewidth=1, label=f'median ({np.median(d):+.3f})')
            ci_lo, ci_hi = percentile_ci(d, 0.95)
            ax.axvline(ci_lo, color='blue', linestyle=':', linewidth=1)
            ax.axvline(ci_hi, color='blue', linestyle=':', linewidth=1)
            ax.set_xlabel(f'{coef} difference')
            ax.set_ylabel('density')
            ax.set_title(f'{pair_label}: Δ{coef}')
            ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'pairwise_differences.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# ==============================================================
# Main driver
# ==============================================================

def main(shared_skel_csv=DEFAULT_SHARED_SKEL_CSV,
         non_nested_csv=DEFAULT_NON_NESTED_CSV,
         nested_csv=DEFAULT_NESTED_CSV,
         shared_skel_stoch_csv=DEFAULT_SHARED_SKEL_STOCH_CSV,
         out_dir='./logs/bootstrap_analysis/',
         n_boot=N_BOOTSTRAPS,
         seed=RANDOM_SEED,
         do_row_bootstrap=True):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("BOOTSTRAP PROTOCOL COMPARISON")
    print("=" * 64)
    print(f"  Bootstraps:  {n_boot}")
    print(f"  Method:      within-cell (primary), row-level (sanity check)")
    print(f"  Random seed: {seed}")
    print()
    print("Loading data...")
    df_shared = load_protocol_csv(shared_skel_csv, 'shared_skel')
    df_nn     = load_protocol_csv(non_nested_csv, 'non_nested')
    df_nested = None
    if nested_csv:
        df_nested = load_protocol_csv(nested_csv, 'nested')
    df_stoch = None
    if shared_skel_stoch_csv:
        if os.path.exists(shared_skel_stoch_csv):
            df_stoch = load_protocol_csv(shared_skel_stoch_csv,
                                         'shared_skel_stoch')
        else:
            print(f"  [shared_skel_stoch CSV not found at "
                  f"{shared_skel_stoch_csv}; continuing without it]")
    print()

    # Original point estimates from the source data (for reference)
    print("Point estimates from the full data (no bootstrap):")
    for label, df in [('shared_skel', df_shared),
                      ('non_nested',  df_nn)]:
        fit = fit_logistic_regression(df)
        print(f"  {label}: alpha={fit['alpha']:+.4f}  "
              f"beta_F={fit['beta_F']:+.4f}  beta_N={fit['beta_N']:+.4f}")
    if df_nested is not None:
        fit = fit_logistic_regression(df_nested)
        print(f"  nested: alpha={fit['alpha']:+.4f}  "
              f"beta_F={fit['beta_F']:+.4f}  beta_N={fit['beta_N']:+.4f}")
    if df_stoch is not None:
        fit = fit_logistic_regression(df_stoch)
        print(f"  shared_skel_stoch: alpha={fit['alpha']:+.4f}  "
              f"beta_F={fit['beta_F']:+.4f}  beta_N={fit['beta_N']:+.4f}")
    print()

    # STEP 1-2: Within-cell bootstrap on each protocol
    print("Step 1: within-cell bootstrap (primary)...")
    rng = np.random.default_rng(seed)
    boot_shared = bootstrap_within_cell(df_shared, n_boot, rng, 'shared_skel')
    rng = np.random.default_rng(seed)
    boot_nn = bootstrap_within_cell(df_nn, n_boot, rng, 'non_nested')
    boot_nested = None
    if df_nested is not None:
        rng = np.random.default_rng(seed)
        boot_nested = bootstrap_within_cell(df_nested, n_boot, rng, 'nested')
    boot_stoch = None
    if df_stoch is not None:
        rng = np.random.default_rng(seed)
        boot_stoch = bootstrap_within_cell(df_stoch, n_boot, rng,
                                           'shared_skel_stoch')

    print("\n--- Within-cell bootstrap CIs ---")
    print_protocol_summary(boot_shared, 'shared_skel')
    print_protocol_summary(boot_nn, 'non_nested')
    if boot_nested is not None:
        print_protocol_summary(boot_nested, 'nested')
    if boot_stoch is not None:
        print_protocol_summary(boot_stoch, 'shared_skel_stoch')

    # Save bootstrap traces
    boot_shared.to_csv(os.path.join(out_dir, 'boot_within_shared_skel.csv'),
                       index=False)
    boot_nn.to_csv(os.path.join(out_dir, 'boot_within_non_nested.csv'),
                   index=False)
    if boot_nested is not None:
        boot_nested.to_csv(os.path.join(out_dir, 'boot_within_nested.csv'),
                           index=False)
    if boot_stoch is not None:
        boot_stoch.to_csv(
            os.path.join(out_dir, 'boot_within_shared_skel_stoch.csv'),
            index=False)

    # STEP 3: Pairwise differences
    print("\nStep 3: pairwise differences (within-cell bootstrap)...")
    diff_results = {}
    diff_ss_nn = compute_pairwise_differences(boot_shared, boot_nn,
                                                'shared_skel', 'non_nested')
    print_pairwise('shared_skel', 'non_nested', diff_ss_nn)
    diff_results['shared_skel_minus_non_nested'] = diff_ss_nn

    if boot_stoch is not None:
        # The fifth protocol's primary comparison: does randomizing the
        # schedule move the surface relative to the deterministic
        # shared-skeleton construction at matched phi targets?
        diff_st_ss = compute_pairwise_differences(
            boot_stoch, boot_shared, 'shared_skel_stoch', 'shared_skel')
        print_pairwise('shared_skel_stoch', 'shared_skel', diff_st_ss)
        diff_results['shared_skel_stoch_minus_shared_skel'] = diff_st_ss

        diff_st_nn = compute_pairwise_differences(
            boot_stoch, boot_nn, 'shared_skel_stoch', 'non_nested')
        print_pairwise('shared_skel_stoch', 'non_nested', diff_st_nn)
        diff_results['shared_skel_stoch_minus_non_nested'] = diff_st_nn

    if boot_nested is not None:
        diff_ss_n = compute_pairwise_differences(boot_shared, boot_nested,
                                                  'shared_skel', 'nested')
        print_pairwise('shared_skel', 'nested', diff_ss_n)
        diff_results['shared_skel_minus_nested'] = diff_ss_n

        diff_nn_n = compute_pairwise_differences(boot_nn, boot_nested,
                                                  'non_nested', 'nested')
        print_pairwise('non_nested', 'nested', diff_nn_n)
        diff_results['non_nested_minus_nested'] = diff_nn_n

        if boot_stoch is not None:
            diff_st_n = compute_pairwise_differences(
                boot_stoch, boot_nested, 'shared_skel_stoch', 'nested')
            print_pairwise('shared_skel_stoch', 'nested', diff_st_n)
            diff_results['shared_skel_stoch_minus_nested'] = diff_st_n
    else:
        # Treat nested coefficients as fixed reference
        print("\n  --- (nested CSV not provided; comparing bootstrap")
        print("       distributions against the nested reference) ---")
        ref_pairs = [('shared_skel', boot_shared),
                     ('non_nested',  boot_nn)]
        if boot_stoch is not None:
            ref_pairs.append(('shared_skel_stoch', boot_stoch))
        for label, boot_df in ref_pairs:
            print(f"\n  --- {label} vs nested reference ---")
            for coef in ['alpha', 'beta_F', 'beta_N']:
                vals = boot_df[coef].values
                ref = NESTED_REFERENCE[coef]
                d = vals - ref
                s = summarize_coefficient(d)
                excl_95 = bool(s['ci95_lo'] > 0 or s['ci95_hi'] < 0)
                marker = '*' if excl_95 else ' '
                print(f"    d{coef:>7}: median={s['median']:+.4f}  "
                      f"95% CI [{s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}] "
                      f"{marker}")

    # STEP 4-5: Plots
    print("\nStep 4-5: plots...")
    boot_dfs = {'shared_skel': boot_shared, 'non_nested': boot_nn}
    if boot_nested is not None:
        boot_dfs['nested'] = boot_nested
    if boot_stoch is not None:
        boot_dfs['shared_skel_stoch'] = boot_stoch
    plot_coefficient_distributions(boot_dfs, out_dir)
    plot_pairwise_differences(diff_results, out_dir, boot_dfs)

    # Optional: row-level bootstrap as sanity check
    if do_row_bootstrap:
        print("\nSanity check: row-level bootstrap...")
        rng = np.random.default_rng(seed)
        boot_shared_row = bootstrap_rows(df_shared, n_boot, rng, 'shared_skel')
        rng = np.random.default_rng(seed)
        boot_nn_row = bootstrap_rows(df_nn, n_boot, rng, 'non_nested')
        boot_stoch_row = None
        if df_stoch is not None:
            rng = np.random.default_rng(seed)
            boot_stoch_row = bootstrap_rows(df_stoch, n_boot, rng,
                                            'shared_skel_stoch')
        print("\n--- Row-level bootstrap CIs (sanity check) ---")
        print_protocol_summary(boot_shared_row, 'shared_skel')
        print_protocol_summary(boot_nn_row, 'non_nested')
        if boot_stoch_row is not None:
            print_protocol_summary(boot_stoch_row, 'shared_skel_stoch')
        print("\n--- Row-level pairwise: shared_skel - non_nested ---")
        diff_row = compute_pairwise_differences(boot_shared_row, boot_nn_row,
                                                  'shared_skel', 'non_nested')
        print_pairwise('shared_skel', 'non_nested', diff_row)
        if boot_stoch_row is not None:
            print("\n--- Row-level pairwise: shared_skel_stoch - "
                  "shared_skel ---")
            diff_row_st = compute_pairwise_differences(
                boot_stoch_row, boot_shared_row,
                'shared_skel_stoch', 'shared_skel')
            print_pairwise('shared_skel_stoch', 'shared_skel', diff_row_st)

    # Save summary JSON
    summary = {
        'n_bootstraps':  n_boot,
        'point_estimates': {
            'shared_skel': fit_logistic_regression(df_shared),
            'non_nested':  fit_logistic_regression(df_nn),
        },
        'within_cell': {
            'shared_skel': {coef: summarize_coefficient(
                              boot_shared[coef].values)
                            for coef in ['alpha', 'beta_F', 'beta_N']},
            'non_nested':  {coef: summarize_coefficient(boot_nn[coef].values)
                            for coef in ['alpha', 'beta_F', 'beta_N']},
        },
        'pairwise_differences': diff_results,
        'nested_reference': NESTED_REFERENCE,
        'flat_reference': FLAT_REFERENCE,
    }
    if df_stoch is not None:
        summary['point_estimates']['shared_skel_stoch'] = \
            fit_logistic_regression(df_stoch)
        summary['within_cell']['shared_skel_stoch'] = {
            coef: summarize_coefficient(boot_stoch[coef].values)
            for coef in ['alpha', 'beta_F', 'beta_N']}
    summary_path = os.path.join(out_dir, 'bootstrap_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    # Final verdict
    print()
    print("=" * 64)
    print("VERDICT")
    print("=" * 64)
    beta_N_diff = diff_ss_nn['beta_N']
    print(f"  shared_skel - non_nested  d(beta_N):")
    print(f"    median:    {beta_N_diff['median']:+.4f}")
    print(f"    95% CI:    [{beta_N_diff['ci95_lo']:+.4f}, "
          f"{beta_N_diff['ci95_hi']:+.4f}]")
    print(f"    99% CI:    [{beta_N_diff['ci99_lo']:+.4f}, "
          f"{beta_N_diff['ci99_hi']:+.4f}]")
    print(f"    p~{beta_N_diff['p_two_sided']:.3f}")
    print()
    if beta_N_diff['excludes_0_95']:
        print("  -> 95% CI EXCLUDES zero: shared_skel beta_N is robustly")
        print("     different from non_nested within current 50 seeds.")
        print("     Replication with new seeds (Option 1) is justified.")
    elif beta_N_diff['p_two_sided'] < 0.20:
        print("  -> Suggestive but not significant at 95%.  The data is")
        print("     consistent with a real effect of moderate size or")
        print("     with no effect at all.  Replication recommended only")
        print("     if the framework story hinges on this distinction.")
    else:
        print("  -> 95% CI INCLUDES zero: the shared_skel vs non_nested")
        print("     gap on beta_N is sampling noise within the current 50")
        print("     seeds.  The framework's bifurcation story (nested")
        print("     vs others) stands; the within-others distinction")
        print("     is not statistically supported.")
        print("     No replication needed.")

    # Fifth-protocol verdict (H3 of the Part II registered program):
    # where does the stochastic-severity surface sit relative to the
    # graduated class and the flat schedule?
    if boot_stoch is not None:
        print()
        print("=" * 64)
        print("VERDICT: STOCHASTIC-SEVERITY PROTOCOL (H3)")
        print("=" * 64)
        d_st = diff_results['shared_skel_stoch_minus_shared_skel']['beta_N']
        print(f"  shared_skel_stoch - shared_skel  d(beta_N):")
        print(f"    median:    {d_st['median']:+.4f}")
        print(f"    95% CI:    [{d_st['ci95_lo']:+.4f}, "
              f"{d_st['ci95_hi']:+.4f}]   p~{d_st['p_two_sided']:.3f}")
        bn = summarize_coefficient(boot_stoch['beta_N'].values)
        print(f"  shared_skel_stoch beta_N: median={bn['median']:+.4f}  "
              f"95% CI [{bn['ci95_lo']:+.4f}, {bn['ci95_hi']:+.4f}]")
        print(f"  Reference bands: graduated class ~[-0.29, -0.21]; "
              f"flat = {FLAT_REFERENCE['beta_N']:+.3f} "
              f"(SE {FLAT_REFERENCE['se_beta_N']:.3f})")
        grad_lo, grad_hi = -0.29, -0.21
        in_grad = (bn['ci95_lo'] <= grad_hi and bn['ci95_hi'] >= grad_lo)
        flat_z = abs(bn['median'] - FLAT_REFERENCE['beta_N']) / max(
            (bn['se'] ** 2 + FLAT_REFERENCE['se_beta_N'] ** 2) ** 0.5, 1e-9)
        if not d_st['excludes_0_95'] and in_grad and flat_z > 2.5:
            print("  -> beta_N indistinguishable from the deterministic")
            print("     shared-skeleton fit, inside the graduated band,")
            print("     and separated from the flat value: the sampled")
            print("     schedule REMAINS ON the graduated-class surface.")
        elif d_st['excludes_0_95'] and flat_z < 2.0:
            print("  -> beta_N separates from the graduated class and is")
            print("     compatible with the flat value: the sampled")
            print("     schedule selects a DIFFERENT surface, in the")
            print("     flat direction (budget-account ordering).")
        else:
            print("  -> Intermediate/ambiguous placement: compare with")
            print("     the T3 dispersion axis before concluding; the")
            print("     kappa sweep orders the accounts directly.")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared_skel', default=DEFAULT_SHARED_SKEL_CSV)
    parser.add_argument('--shared_skel_stoch', default=DEFAULT_SHARED_SKEL_STOCH_CSV)
    parser.add_argument('--non_nested',  default=DEFAULT_NON_NESTED_CSV)
    parser.add_argument('--nested',      default=DEFAULT_NESTED_CSV,
                        help='Optional nested CSV (uses NESTED_REFERENCE '
                             'point estimates if omitted)')
    parser.add_argument('--out_dir', default='./logs/bootstrap_protocol_comparison/')
    parser.add_argument('--n_boot',  type=int, default=N_BOOTSTRAPS)
    parser.add_argument('--seed',    type=int, default=RANDOM_SEED)
    parser.add_argument('--skip_row_bootstrap', action='store_true')
    args = parser.parse_args()
    summary = main(shared_skel_csv=args.shared_skel,
         non_nested_csv=args.non_nested,
         nested_csv=args.nested,
         shared_skel_stoch_csv=args.shared_skel_stoch,
         out_dir=args.out_dir,
         n_boot=args.n_boot,
         seed=args.seed,
         do_row_bootstrap=not args.skip_row_bootstrap)
    
    print(summary)

    print()
