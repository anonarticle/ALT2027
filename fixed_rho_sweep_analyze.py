"""
analyze_fixed_rho_sweep.py

Reproduces the analysis I did on the 40x20 sweep CSV, producing a
report (markdown) plus a small set of diagnostic plots.

Usage
-----
    python analyze_fixed_rho_sweep.py path/to/fixed_rho_sweep.csv \
        --report   path/to/report.md \
        --plot-dir path/to/plots/

If --report or --plot-dir are omitted, defaults are placed next to
the input CSV.

What it computes
----------------
  1. Sanity check: F*-density behaviour vs N (catches family-builder
     bugs from the previous run).
  2. P(FI), P(coher fail), P(topo fail) for every (mode, rho, N) cell.
  3. The decisive Plot-B test: at matched F*-density (with N>=2),
     does P(failure) depend on rho independently?  This is the test
     that distinguishes "geometry alone" from "voting margin matters".
  4. Identification of the binding mechanism in each F*-density
     window (coherence vs topology vs free).
  5. Comparison of iid vs shared_skel modes.
  6. Plots: F*-density vs N (per rho), Plot A (P(FI) vs N), Plot B
     (P(FI) vs F*-density with rho colouring).

The report flags conclusions and caveats automatically based on the
data, so it works on any sweep CSV with the expected columns.
"""

import argparse
import os
import sys
from textwrap import dedent

import numpy as np
import pandas as pd

# Constants used throughout the analysis
P_C_SITE = 0.5927       # 2D square lattice site-percolation threshold
P_C_BOND = 0.5          # 2D square lattice bond-percolation threshold
F_BIN_WIDTH = 0.05      # bin width for F*-density grouping
SAMPLING_NOISE_THRESH = 0.10   # spreads below this are "within noise"
RESOLVABLE_VOTING_THRESH = 0.15  # spreads above this resolve voting effect


# ===================================================================
# Loading & preprocessing
# ===================================================================

REQUIRED_COLS = {
    'mode', 'rho', 'N', 'seed',
    'F_star_size', 'F_star_density', 'C_V', 'C_interp',
}


def load_sweep(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )
    df['fi_holds']   = df['C_interp'].isin(['marginal', 'free'])
    df['fail_topo']  = df['C_interp'] == 'no_path'
    df['fail_coher'] = df['C_interp'] == 'incoherent'
    return df


# ===================================================================
# Section 1: F*-density sanity check
# ===================================================================

def section_1_f_star_sanity(df: pd.DataFrame) -> dict:
    """
    Did the family-builder produce envs whose intersection actually
    shrinks with N?  We compare observed F*-density to the i.i.d.
    prediction (1-rho)^N and report the ratio.

    Returns a dict with structured findings for the report.
    """
    out = {}
    pivots = {}
    structural_ratios = {}

    for mode, gmode in df.groupby('mode'):
        rows = []
        for (rho, N), gcell in gmode.groupby(['rho', 'N']):
            obs = gcell['F_star_density'].mean()
            pred = (1 - rho) ** N
            ratio = obs / max(pred, 1e-12)
            rows.append({
                'rho': rho, 'N': N,
                'F_star_density_obs': obs,
                'F_star_density_std': gcell['F_star_density'].std(),
                'F_star_density_pred_iid': pred,
                'ratio_obs_over_iid': ratio,
            })
        pivots[mode] = pd.DataFrame(rows).sort_values(['rho', 'N'])

        # The headline ratio: max ratio observed (worst structural correlation)
        max_ratio_idx = pivots[mode]['ratio_obs_over_iid'].idxmax()
        max_row = pivots[mode].loc[max_ratio_idx]
        structural_ratios[mode] = {
            'max_ratio': max_row['ratio_obs_over_iid'],
            'rho_at_max': max_row['rho'],
            'N_at_max': int(max_row['N']),
        }

        # Verify F* shrinks: largest N in the smallest rho should be
        # well below F*-density at N=1.
        # min_rho = pivots[mode]['rho'].min()
        # min_N = pivots[mode]['N'].min()
        # N_max = pivots[mode][pivots[mode]['rho'] == min_rho]['N'].max()

        min_N, min_rho = min(zip(pivots[mode]['N'].values, pivots[mode]['rho'].values))
        N_max, _ = max(zip(pivots[mode]['N'].values, pivots[mode]['rho'].values))
        min_rho_at_N_max = pivots[mode][pivots[mode]['N'] == N_max]['rho'].min()
        f_at_1 = pivots[mode][(pivots[mode]['rho'] == min_rho) &
                              (pivots[mode]['N'] == min_N)]['F_star_density_obs'].iloc[0]
        f_at_max = pivots[mode][(pivots[mode]['rho'] == min_rho_at_N_max) &
                                (pivots[mode]['N'] == N_max)]['F_star_density_obs'].iloc[0]
        structural_ratios[mode]['f_drop_min_rho'] = f_at_1 - f_at_max

    out['pivots'] = pivots
    out['structural_ratios'] = structural_ratios
    out['family_builder_ok'] = all(
        v['f_drop_min_rho'] > 0.05 for v in structural_ratios.values()
    )
    return out


# ===================================================================
# Section 2: per-cell breakdown of failure modes
# ===================================================================

def section_2_per_cell(df: pd.DataFrame) -> dict:
    out = {}
    cells = (df.groupby(['mode', 'rho', 'N'])
               .agg(p_fi=('fi_holds', 'mean'),
                    p_topo=('fail_topo', 'mean'),
                    p_coher=('fail_coher', 'mean'),
                    cv_mean=('C_V', 'mean'),
                    cv_median=('C_V', 'median'),
                    F_dens=('F_star_density', 'mean'),
                    n=('seed', 'count'))
               .reset_index())
    out['cells'] = cells
    out['n_per_cell'] = cells['n'].iloc[0] if len(cells) else 0

    # Failure mode totals (overall mix in the sweep)
    out['mix'] = df['C_interp'].value_counts().to_dict()
    out['has_free_outcomes'] = bool((df['C_V'] > 0).any())
    return out


# ===================================================================
# Section 3: the decisive Plot-B test
# ===================================================================

def section_3_plot_b_collapse(df: pd.DataFrame) -> dict:
    """
    Bin F*-density and look for cross-rho variation in P(failure)
    *within* each bin, restricted to N>=2 (so coherence can fail
    meaningfully).  This is the discriminator between Scenario 1
    (geometry suffices) and Scenario 2 (voting margin matters
    independently).
    """
    out = {}
    bins = np.arange(0.30, 1.01, F_BIN_WIDTH)

    # We focus on iid mode since that's where the percolation prediction
    # is cleanest.  shared_skel is reported as a sanity comparison.
    spreads_per_mode = {}
    bin_tables_per_mode = {}

    for mode in df['mode'].unique():
        sub = df[(df['mode'] == mode) & (df['N'] >= 2)].copy()
        sub['F_bin'] = pd.cut(sub['F_star_density'], bins=bins,
                              include_lowest=True)
        bin_table = (sub.groupby(['F_bin', 'rho'], observed=True)
                        .agg(n=('fi_holds', 'count'),
                             N_mean=('N', 'mean'),
                             p_fi=('fi_holds', 'mean'),
                             p_coher=('fail_coher', 'mean'),
                             p_topo=('fail_topo', 'mean'))
                        .reset_index())
        bin_table = bin_table[bin_table['n'] >= 3]   # need >=3 obs per bin/rho
        bin_tables_per_mode[mode] = bin_table

        # Find bins with >=2 distinct rho values — the discriminating data.
        bin_rho_counts = (bin_table.groupby('F_bin', observed=True)
                                   .size().reset_index(name='n_rhos'))
        contested_bins = bin_rho_counts[bin_rho_counts['n_rhos'] >= 2]['F_bin']

        # Compute spread per contested bin
        spread_rows = []
        for fb in contested_bins:
            cell = bin_table[bin_table['F_bin'] == fb]
            rhos = sorted(cell['rho'].unique())
            p_fi_vals = cell.set_index('rho').loc[rhos, 'p_fi'].values
            p_coher_vals = cell.set_index('rho').loc[rhos, 'p_coher'].values
            p_topo_vals = cell.set_index('rho').loc[rhos, 'p_topo'].values
            spread_rows.append({
                'F_bin': fb,
                'rhos': rhos,
                'p_fi': list(p_fi_vals),
                'p_fi_spread': float(p_fi_vals.max() - p_fi_vals.min()),
                'p_coher': list(p_coher_vals),
                'p_coher_spread': float(p_coher_vals.max() - p_coher_vals.min()),
                'p_topo': list(p_topo_vals),
                'p_topo_spread': float(p_topo_vals.max() - p_topo_vals.min()),
                'n_per_cell': int(cell['n'].mean()),
            })
        spreads_per_mode[mode] = pd.DataFrame(spread_rows)

    out['bin_tables'] = bin_tables_per_mode
    out['spreads'] = spreads_per_mode

    # Pick a "primary mode" for the headline verdict: prefer 'iid'
    # since that's where the percolation prediction is cleanest, but
    # fall back to whatever's available so single-mode CSVs work.
    available_modes = [m for m in spreads_per_mode
                       if len(spreads_per_mode[m]) > 0]
    if 'iid' in available_modes:
        primary_mode = 'iid'
    elif available_modes:
        primary_mode = available_modes[0]
    else:
        primary_mode = None
    out['primary_mode'] = primary_mode

    # Headline numbers for the report (focus on primary mode)
    if primary_mode is not None:
        ss = spreads_per_mode[primary_mode]
        out['max_p_fi_spread'] = ss['p_fi_spread'].max()
        out['max_p_coher_spread'] = ss['p_coher_spread'].max()
        out['max_p_topo_spread'] = ss['p_topo_spread'].max()
        out['n_per_cell_typical'] = int(ss['n_per_cell'].median())

        # Sampling SE for a binomial proportion at this n
        n = max(out['n_per_cell_typical'], 1)
        out['se_proportion'] = float(np.sqrt(0.25 / n))

        # Sample-size-aware thresholds:
        #   - "Within noise" means cross-rho spread is below 2*SE,
        #     i.e. the difference of two binomial proportions could
        #     plausibly arise from sampling alone.
        #   - "Resolvable voting effect" means spread is above 4*SE,
        #     i.e. a real underlying difference of >= 2*SE.
        # At n=10 these are 0.32 and 0.63; at n=50, 0.14 and 0.28.
        noise_thresh = 2.0 * out['se_proportion']
        resolvable_thresh = 4.0 * out['se_proportion']
        out['noise_thresh'] = noise_thresh
        out['resolvable_thresh'] = resolvable_thresh

        # Verdict
        if out['max_p_fi_spread'] <= noise_thresh:
            out['verdict'] = 'geometry_sufficient'
        elif out['max_p_fi_spread'] >= resolvable_thresh:
            out['verdict'] = 'voting_resolvable'
        else:
            out['verdict'] = 'inconclusive'
    else:
        out['verdict'] = 'no_data'

    return out


# ===================================================================
# Section 4: binding-mechanism map (coherence vs topology vs free)
# ===================================================================

def section_4_binding_mechanism(df: pd.DataFrame, primary_mode: str) -> dict:
    """
    For each F*-density window (primary_mode, all N>=2), report the
    dominant binding mechanism: free, coherence, or topology.
    """
    out = {'primary_mode': primary_mode}
    if primary_mode is None:
        out['table'] = pd.DataFrame()
        out['crossover_window'] = None
        out['p_c_site'] = P_C_SITE
        out['p_c_bond'] = P_C_BOND
        return out

    bins = np.arange(0.30, 1.01, F_BIN_WIDTH)

    sub = df[(df['mode'] == primary_mode) & (df['N'] >= 2)].copy()
    sub['F_bin'] = pd.cut(sub['F_star_density'], bins=bins,
                          include_lowest=True)
    g = (sub.groupby('F_bin', observed=True)
             .agg(p_fi=('fi_holds', 'mean'),
                  p_coher=('fail_coher', 'mean'),
                  p_topo=('fail_topo', 'mean'),
                  n=('fi_holds', 'count'))
             .reset_index())

    def classify_row(r):
        if r['p_fi'] >= 0.5:
            return 'free_or_marginal'
        if r['p_coher'] > r['p_topo']:
            return 'coherence_binds'
        if r['p_topo'] > r['p_coher']:
            return 'topology_binds'
        return 'tied'

    g['binding'] = g.apply(classify_row, axis=1)
    out['table'] = g

    # Locate the crossover from coherence-binds to topology-binds
    # (the F*-density at which they swap dominance).
    coher_bins = g[g['binding'] == 'coherence_binds']
    topo_bins = g[g['binding'] == 'topology_binds']
    if len(coher_bins) > 0 and len(topo_bins) > 0:
        # Lowest F*-density at which coherence still binds, vs highest
        # F*-density at which topology binds.  Crossover sits between.
        coher_low = float(coher_bins['F_bin'].apply(lambda b: b.left).min())
        topo_high = float(topo_bins['F_bin'].apply(lambda b: b.right).max())
        out['crossover_window'] = (min(coher_low, topo_high),
                                   max(coher_low, topo_high))
    else:
        out['crossover_window'] = None

    # Compare crossover to percolation thresholds.
    out['p_c_site'] = P_C_SITE
    out['p_c_bond'] = P_C_BOND
    return out


# ===================================================================
# Section 5: iid vs shared_skel comparison
# ===================================================================

def section_5_iid_vs_shared_skel(df: pd.DataFrame) -> dict:
    out = {}
    if not {'iid', 'shared_skel'}.issubset(df['mode'].unique()):
        out['both_modes_present'] = False
        return out
    out['both_modes_present'] = True

    cells = (df.groupby(['rho', 'N', 'mode'])
               .agg(p_fi=('fi_holds', 'mean'),
                    f_dens=('F_star_density', 'mean'))
               .reset_index())
    piv_pfi = cells.pivot_table(index=['rho', 'N'], columns='mode',
                                values='p_fi')
    piv_fd = cells.pivot_table(index=['rho', 'N'], columns='mode',
                               values='f_dens')

    diff_pfi = piv_pfi['iid'] - piv_pfi['shared_skel']
    diff_fd = piv_fd['iid'] - piv_fd['shared_skel']
    out['piv_pfi'] = piv_pfi
    out['piv_fd'] = piv_fd
    out['max_abs_diff_pfi'] = float(diff_pfi.abs().max())
    out['mean_abs_diff_pfi'] = float(diff_pfi.abs().mean())
    out['mean_diff_fd'] = float(diff_fd.mean())   # iid usually has lower F*-dens
    return out


# ===================================================================
# Plots
# ===================================================================

def make_plots(df: pd.DataFrame, plot_dir: str) -> list:
    """Generate the standard diagnostic plots, return list of paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    plot_paths = []

    rhos = sorted(df['rho'].unique())
    modes = sorted(df['mode'].unique())
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(rhos)))
    rho_colour = dict(zip(rhos, colours))

    # ----- Plot 1: F*-density vs N, per (mode, rho), with iid prediction
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4),
                             sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        sub = df[df['mode'] == mode]
        for rho in rhos:
            g = (sub[sub['rho'] == rho]
                 .groupby('N')['F_star_density']
                 .agg(['mean', 'std'])
                 .reset_index())
            ax.errorbar(g['N'], g['mean'], yerr=g['std'],
                        marker='o', label=f'rho={rho:.2f}',
                        color=rho_colour[rho], capsize=3)
            # iid prediction
            Ns_dense = np.linspace(g['N'].min(), g['N'].max(), 50)
            ax.plot(Ns_dense, (1 - rho) ** Ns_dense,
                    color=rho_colour[rho], linestyle='--', alpha=0.4)
        ax.axhline(P_C_SITE, color='red', linestyle=':',
                   label=f'$p_c^{{site}}$={P_C_SITE:.4f}')
        ax.axhline(P_C_BOND, color='red', linestyle='-.',
                   label=f'$p_c^{{bond}}$={P_C_BOND:.2f}', alpha=0.5)
        ax.set_xlabel('N')
        ax.set_ylabel(r'$|F^*| / |\mathcal{S}|$')
        ax.set_title(f'mode = {mode}')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle('F*-density vs N  (solid = empirical, dashed = i.i.d. prediction)')
    fig.tight_layout()
    p = os.path.join(plot_dir, '01_f_star_vs_N.png')
    fig.savefig(p, dpi=120)
    plot_paths.append(p)
    plt.close(fig)

    # ----- Plot 2: Plot A - P(FI) vs N per rho, per mode
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4),
                             sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        sub = df[df['mode'] == mode]
        for rho in rhos:
            g = (sub[sub['rho'] == rho]
                 .groupby('N')['fi_holds']
                 .mean().reset_index())
            ax.plot(g['N'], g['fi_holds'], marker='o',
                    color=rho_colour[rho], label=f'rho={rho:.2f}')
        ax.set_xlabel('N')
        ax.set_ylabel('P(FI holds)')
        ax.set_title(f'mode = {mode}')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle('Plot A: Free Inference success vs N')
    fig.tight_layout()
    p = os.path.join(plot_dir, '02_plot_A_p_fi_vs_N.png')
    fig.savefig(p, dpi=120)
    plot_paths.append(p)
    plt.close(fig)

    # ----- Plot 3: Plot B - P(FI) vs F*-density, coloured by rho
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4),
                             sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        sub = df[(df['mode'] == mode) & (df['N'] >= 2)]
        for rho in rhos:
            sr = sub[sub['rho'] == rho]
            # bin F*-density and report mean P(FI) per bin
            bins = np.arange(0.30, 1.01, F_BIN_WIDTH)
            sr = sr.copy()
            sr['F_bin'] = pd.cut(sr['F_star_density'], bins=bins,
                                 include_lowest=True)
            g = (sr.groupby('F_bin', observed=True)
                   .agg(p_fi=('fi_holds', 'mean'),
                        f=('F_star_density', 'mean'),
                        n=('fi_holds', 'count'))
                   .reset_index())
            g = g[g['n'] >= 3]
            if len(g):
                ax.plot(g['f'], g['p_fi'], marker='o',
                        color=rho_colour[rho], label=f'rho={rho:.2f}')
        ax.axvline(P_C_SITE, color='red', linestyle=':',
                   label=f'$p_c^{{site}}$={P_C_SITE:.4f}')
        ax.axvline(P_C_BOND, color='red', linestyle='-.',
                   label=f'$p_c^{{bond}}$={P_C_BOND:.2f}', alpha=0.5)
        ax.set_xlabel(r'$|F^*| / |\mathcal{S}|$')
        ax.set_ylabel('P(FI holds)')
        ax.set_title(f'mode = {mode}')
        ax.set_xlim(0.30, 1.0)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle('Plot B: P(FI) vs F*-density.  Curves collapsing across rho => geometry sufficient.')
    fig.tight_layout()
    p = os.path.join(plot_dir, '03_plot_B_collapse.png')
    fig.savefig(p, dpi=120)
    plot_paths.append(p)
    plt.close(fig)

    # ----- Plot 4: failure-mode decomposition vs F*-density (primary mode)
    # Prefer 'iid' if present, otherwise fall back to whatever exists.
    plot4_mode = 'iid' if 'iid' in modes else (modes[0] if modes else None)
    if plot4_mode is not None:
        sub = df[(df['mode'] == plot4_mode) & (df['N'] >= 2)].copy()
        bins = np.arange(0.30, 1.01, F_BIN_WIDTH)
        sub['F_bin'] = pd.cut(sub['F_star_density'], bins=bins,
                              include_lowest=True)
        g = (sub.groupby('F_bin', observed=True)
                .agg(p_free=('fi_holds', 'mean'),
                     p_coher=('fail_coher', 'mean'),
                     p_topo=('fail_topo', 'mean'),
                     f=('F_star_density', 'mean'),
                     n=('fi_holds', 'count'))
                .reset_index())
        g = g[g['n'] >= 3].sort_values('f')

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(g['f'], 0, g['p_topo'], alpha=0.6,
                        label='P(topology fail)', color='#d62728')
        ax.fill_between(g['f'], g['p_topo'], g['p_topo'] + g['p_coher'],
                        alpha=0.6, label='P(coherence fail)', color='#ff7f0e')
        ax.fill_between(g['f'], g['p_topo'] + g['p_coher'], 1.0,
                        alpha=0.6, label='P(FI holds)', color='#2ca02c')
        ax.axvline(P_C_SITE, color='black', linestyle=':',
                   label=f'$p_c^{{site}}$')
        ax.axvline(P_C_BOND, color='black', linestyle='-.',
                   label=f'$p_c^{{bond}}$', alpha=0.5)
        ax.set_xlabel(r'$|F^*| / |\mathcal{S}|$')
        ax.set_ylabel('Probability')
        ax.set_title(f'Failure-mode decomposition vs F*-density '
                     f'({plot4_mode}, N>=2)')
        ax.set_xlim(0.30, 1.0)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc='center left')
        fig.tight_layout()
        p = os.path.join(plot_dir, '04_failure_modes_stacked.png')
        fig.savefig(p, dpi=120)
        plot_paths.append(p)
        plt.close(fig)

    return plot_paths


# ===================================================================
# Report writer
# ===================================================================

def write_report(s1, s2, s3, s4, s5, plot_paths, csv_path, report_path):
    """Assemble the markdown report from the section results."""
    csv_name = os.path.basename(csv_path)
    plot_dir_rel = (os.path.relpath(os.path.dirname(plot_paths[0]),
                                    os.path.dirname(report_path))
                    if plot_paths else '.')

    # ---- 1. Header --------------------------------------------------
    md = []
    md.append(f"# Sweep analysis: `{csv_name}`")
    md.append("")
    md.append(f"**Cells analysed:** {len(s2['cells'])} unique (mode, rho, N) "
              f"with n={s2['n_per_cell']} seeds each.")
    md.append("")
    md.append(f"**Failure mix overall:** {s2['mix']}.  "
              f"{'Some cells achieved C_V > 0 (positive margin).' if s2['has_free_outcomes'] else 'No cells achieved C_V > 0; the best outcome is `marginal` (C_V = 0).'}")
    md.append("")

    # ---- 2. Headline verdict -----------------------------------------
    md.append("## Headline verdict")
    md.append("")
    if 'verdict' in s3:
        verdict = s3['verdict']
        n = s2['n_per_cell']
        se = s3.get('se_proportion', 0.0)
        nt = s3.get('noise_thresh', SAMPLING_NOISE_THRESH)
        rt = s3.get('resolvable_thresh', RESOLVABLE_VOTING_THRESH)
        primary_mode = s3.get('primary_mode')
        max_spread = s3.get('max_p_fi_spread')

        if primary_mode is None or max_spread is None:
            md.append("> **No data available for the headline test** — "
                      "no F*-density bins had multiple rho values present, "
                      "so the cross-rho spread cannot be computed. "
                      "Sweep needs at least 2 distinct rho values with "
                      "overlapping F*-density bins.")
        else:
            mode_note = (f" *(based on `{primary_mode}` mode; "
                         + ("the mode where the percolation prediction is "
                            "cleanest)*" if primary_mode == 'iid'
                            else "the only mode in the sweep)*"))
            md.append(f"*Sample-size context: n={n} seeds per cell, binomial "
                      f"SE ≈ {se:.2f}.  Spreads below {nt:.2f} (≈2·SE) are "
                      f"within sampling noise; spreads above {rt:.2f} (≈4·SE) "
                      f"are statistically resolvable.*")
            md.append("")
            if verdict == 'geometry_sufficient':
                md.append("> **Geometry is sufficient.**" + mode_note + " "
                          f"Maximum cross-rho spread in P(FI) at matched "
                          f"F*-density is {max_spread:.2f}, "
                          f"below the noise threshold ({nt:.2f}). "
                          f"At fixed F*-density, success rate does not depend "
                          f"on rho. This falsifies the strong form of the "
                          f"voting argument: the value mixture's per-cell "
                          f"decisions reduce to a function of consensus "
                          f"geometry alone on this generator.")
            elif verdict == 'voting_resolvable':
                md.append("> **Voting margin matters beyond geometry.**" + mode_note + " "
                          f"Maximum cross-rho spread in P(FI) at matched "
                          f"F*-density is {max_spread:.2f}, "
                          f"above the resolvable threshold ({rt:.2f}). "
                          f"At fixed F*-density, success rate depends on rho "
                          f"even after geometry is held fixed.  The voting "
                          f"mechanism is operating as an independent constraint.")
            else:
                md.append("> **Inconclusive at current resolution.**" + mode_note + " "
                          f"Maximum cross-rho spread in P(FI) at matched "
                          f"F*-density is {max_spread:.2f}, "
                          f"between noise threshold ({nt:.2f}) and "
                          f"resolvable threshold ({rt:.2f}). "
                          f"At n={n} per cell the data is consistent with "
                          f"both 'geometry sufficient' and a real voting "
                          f"effect of size up to ~2·SE.  "
                          f"Re-run with n≥50 to push the noise threshold "
                          f"below 0.15.")
    md.append("")

    # ---- 3. Section 1: F*-density sanity ------------------------------
    md.append("## 1. F*-density vs N — family-builder sanity")
    md.append("")
    if s1['family_builder_ok']:
        md.append("**Family-builder is producing genuinely independent envs.** "
                  "F*-density drops substantially with N at every rho.")
    else:
        md.append("⚠ **Possible family-builder bug**: F*-density does not drop "
                  "appreciably with N.  All N envs may be near-identical.")
    md.append("")
    for mode, piv in s1['pivots'].items():
        md.append(f"### Mode = `{mode}`")
        md.append("")
        md.append("| rho | N | F\\*-density observed | (1-rho)^N predicted | ratio |")
        md.append("|----:|----:|----:|----:|----:|")
        for _, r in piv.iterrows():
            md.append(f"| {r['rho']:.2f} | {int(r['N'])} | "
                      f"{r['F_star_density_obs']:.4f} ± {r['F_star_density_std']:.4f} | "
                      f"{r['F_star_density_pred_iid']:.4f} | "
                      f"{r['ratio_obs_over_iid']:.2f}× |")
        md.append("")
        sr = s1['structural_ratios'][mode]
        md.append(f"Maximum ratio of observed to i.i.d. prediction: "
                  f"**{sr['max_ratio']:.2f}×** at "
                  f"(rho={sr['rho_at_max']:.2f}, N={sr['N_at_max']}). "
                  f"Ratios well above 1 indicate generator-level structural "
                  f"correlation; the maze generator's DFS-carved corridors "
                  f"share more wall positions across seeds than fully i.i.d. "
                  f"obstacles would.")
        md.append("")

    # ---- 4. Section 2: per-cell breakdown -----------------------------
    md.append("## 2. Per-cell P(FI), P(coher fail), P(topo fail)")
    md.append("")
    cells = s2['cells']
    for mode in sorted(cells['mode'].unique()):
        md.append(f"### Mode = `{mode}`")
        md.append("")
        md.append("| rho | N | F\\*-density | P(FI) | P(topo) | P(coher) |")
        md.append("|----:|----:|----:|----:|----:|----:|")
        for _, r in cells[cells['mode'] == mode].sort_values(['rho', 'N']).iterrows():
            md.append(f"| {r['rho']:.2f} | {int(r['N'])} | "
                      f"{r['F_dens']:.3f} | "
                      f"{r['p_fi']:.2f} | {r['p_topo']:.2f} | {r['p_coher']:.2f} |")
        md.append("")

    # ---- 5. Section 3: Plot-B test -----------------------------------
    md.append("## 3. The decisive test: Plot-B collapse")
    md.append("")
    md.append("At matched F*-density (with N≥2 so coherence can fail "
              "meaningfully), we ask: *does P(failure) depend on rho "
              "independently of geometry?*  If yes, voting margin is "
              "doing work beyond geometry; if no, geometry suffices.")
    md.append("")
    primary_mode = s3.get('primary_mode')
    if primary_mode is None:
        md.append("*No mode has bins with multiple rho values; cannot "
                  "perform the test.*")
    else:
        primary_label = ("iid mode (the cleanest test, "
                         "where the percolation prediction is sharpest)"
                         if primary_mode == 'iid'
                         else f"`{primary_mode}` mode (the only mode in this sweep)")
        md.append(f"### Test: {primary_label}")
        md.append("")
        primary_spreads = s3['spreads'].get(primary_mode)
        if primary_spreads is None or len(primary_spreads) == 0:
            md.append("*No bins with multiple rho values present; cannot "
                      "perform the test.*")
        else:
            md.append("| F\\*-bin | rhos present | P(FI) values | spread |")
            md.append("|:----:|:----:|:----:|----:|")
            for _, r in primary_spreads.iterrows():
                md.append(f"| {r['F_bin']} | "
                          f"{[f'{x:.2f}' for x in r['rhos']]} | "
                          f"{[f'{x:.2f}' for x in r['p_fi']]} | "
                          f"{r['p_fi_spread']:.2f} |")
            md.append("")
            md.append(f"**Maximum cross-rho spread in P(FI):** "
                      f"{s3['max_p_fi_spread']:.2f}.  "
                      f"**Sampling SE at n={s2['n_per_cell']}:** "
                      f"{s3['se_proportion']:.2f}.")
            md.append("")
            md.append(f"Decomposed by failure mode: max spread in "
                      f"P(coher fail) = {s3['max_p_coher_spread']:.2f}; "
                      f"in P(topo fail) = {s3['max_p_topo_spread']:.2f}.")
    md.append("")

    # If a second mode is also present, show its spreads too (descriptive only)
    other_modes = [m for m in s3.get('spreads', {})
                   if m != primary_mode and len(s3['spreads'][m]) > 0]
    for om in other_modes:
        md.append(f"### Cross-check: `{om}` mode")
        md.append("")
        os_table = s3['spreads'][om]
        md.append("| F\\*-bin | rhos present | P(FI) values | spread |")
        md.append("|:----:|:----:|:----:|----:|")
        for _, r in os_table.iterrows():
            md.append(f"| {r['F_bin']} | "
                      f"{[f'{x:.2f}' for x in r['rhos']]} | "
                      f"{[f'{x:.2f}' for x in r['p_fi']]} | "
                      f"{r['p_fi_spread']:.2f} |")
        md.append("")

    # ---- 6. Section 4: binding mechanism ------------------------------
    primary_mode = s4.get('primary_mode')
    mode_str = f"`{primary_mode}`" if primary_mode else "primary mode"
    md.append(f"## 4. Binding mechanism by F*-density window ({mode_str}, N≥2)")
    md.append("")
    if primary_mode is None or len(s4['table']) == 0:
        md.append("*No primary-mode data with N≥2 available; section skipped.*")
        md.append("")
    else:
        md.append("| F\\*-bin | P(FI) | P(coher fail) | P(topo fail) | binding mechanism |")
        md.append("|:----:|----:|----:|----:|:----:|")
        for _, r in s4['table'].sort_values('F_bin').iterrows():
            md.append(f"| {r['F_bin']} | {r['p_fi']:.2f} | "
                      f"{r['p_coher']:.2f} | {r['p_topo']:.2f} | "
                      f"{r['binding']} |")
        md.append("")
        if s4['crossover_window'] is not None:
            a, b = s4['crossover_window']
            md.append(f"**Coherence → topology crossover** sits in the F*-density "
                      f"window ≈ ({a:.2f}, {b:.2f}).")
        md.append("")
    md.append(f"**Percolation reference:** "
              f"$p_c^{{site}} = {P_C_SITE:.4f}$, "
              f"$p_c^{{bond}} = {P_C_BOND:.4f}$.  Topological failures "
              f"are predicted to dominate as F*-density falls below "
              f"these values.")
    md.append("")

    # ---- 7. Section 5: iid vs shared_skel -----------------------------
    md.append("## 5. iid vs shared_skel comparison")
    md.append("")
    if not s5.get('both_modes_present'):
        md.append("*Only one mode present in the sweep; comparison "
                  "skipped.*")
    else:
        md.append(f"Maximum |P(FI)_iid − P(FI)_shared_skel| across cells: "
                  f"**{s5['max_abs_diff_pfi']:.2f}**; mean: "
                  f"{s5['mean_abs_diff_pfi']:.2f}.")
        md.append("")
        md.append(f"Mean F*-density difference (iid − shared_skel): "
                  f"{s5['mean_diff_fd']:+.3f}.  "
                  f"{'shared_skel has higher F*-density (more shared free cells).' if s5['mean_diff_fd'] < 0 else 'iid has higher F*-density.'}")
        md.append("")
        md.append("| (rho, N) | iid P(FI) | shared_skel P(FI) | diff |")
        md.append("|:----:|----:|----:|----:|")
        for idx, row in s5['piv_pfi'].iterrows():
            diff = row['iid'] - row['shared_skel']
            md.append(f"| {idx} | {row['iid']:.2f} | "
                      f"{row['shared_skel']:.2f} | {diff:+.2f} |")
    md.append("")

    # ---- 8. Plots ---------------------------------------------------
    if plot_paths:
        md.append("## Plots")
        md.append("")
        for p in plot_paths:
            rel = os.path.join(plot_dir_rel, os.path.basename(p))
            md.append(f"![{os.path.basename(p)}]({rel})")
            md.append("")

    # ---- 9. Recommended next steps ----------------------------------
    md.append("## Recommended next steps")
    md.append("")
    if s3.get('verdict') == 'inconclusive':
        md.append("- **Re-run with n≥50 per cell** to resolve the "
                  "matched-F* spread.  At current n the spread is "
                  "between the noise floor and the resolvable threshold.")
    if s1['family_builder_ok'] and any(
            v['max_ratio'] > 5.0 for v in s1['structural_ratios'].values()):
        md.append("- **Repeat on a larger grid** (e.g., 100×50) to "
                  "reduce maze-generator structural correlation.  Current "
                  "ratio of observed F*-density to i.i.d. prediction is "
                  "substantial, suggesting generator-level effects "
                  "inflate the consensus density.")
    if not s2['has_free_outcomes']:
        md.append("- **Note:** No cells achieved positive C_V (strict "
                  "Free Inference); only `marginal` (C_V=0) is reached. "
                  "If you want to study the strict-FI regime, the Q-table "
                  "training or the family-construction needs to admit "
                  "positive margins.")
    md.append("- **For a targeted bps test,** run rollouts only in the "
              "F*-density window where coherence binds (from section 4) "
              "and check whether VM bps depends on rho at fixed F*-density. "
              "This is a more sensitive test of the voting mechanism than "
              "P(FI).")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    return

def analyze_results(csv_path):

    base = os.path.splitext(csv_path)[0]
    report_path = f"{base}_report.md"
    plot_dir = f"{base}_plots"

    print(f"Loading {csv_path}...")
    df = load_sweep(csv_path)
    print(f"  Loaded {len(df)} rows, modes={sorted(df['mode'].unique())}, "
          f"rhos={sorted(df['rho'].unique())}, "
          f"Ns={sorted(df['N'].unique())}.")

    print("Running section 1: F*-density sanity check...")
    s1 = section_1_f_star_sanity(df)
    print("Running section 2: per-cell breakdown...")
    s2 = section_2_per_cell(df)
    print("Running section 3: Plot-B collapse test...")
    s3 = section_3_plot_b_collapse(df)
    print("Running section 4: binding mechanism map...")
    s4 = section_4_binding_mechanism(df, primary_mode=s3.get('primary_mode'))
    print("Running section 5: iid vs shared_skel...")
    s5 = section_5_iid_vs_shared_skel(df)

    print(f"Generating plots in {plot_dir}/...")
    try:
        plots = make_plots(df, plot_dir)
        print(f"  Wrote {len(plots)} plot(s).")
    except Exception as exc:
        print(f"  Plot generation failed: {exc}")
        plots = []

    print(f"Writing report to {report_path}...")
    write_report(s1, s2, s3, s4, s5, plots, csv_path, report_path)

    print()
    print(f"Done.  Report: {report_path}")
    if plots:
        print(f"Plots:  {plot_dir}/")
    if 'verdict' in s3:
        print(f"Verdict: {s3['verdict']}.")
            
    return 0

# ===================================================================
# Driver
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse a fixed_rho_sweep CSV and emit a report.")
    parser.add_argument('csv_path',
                        help="Path to fixed_rho_sweep CSV file.")
    parser.add_argument('--report', default=True,
                        help="Output markdown report path "
                             "(default: alongside CSV with .md suffix).")
    parser.add_argument('--plot-dir', default=True,
                        help="Output directory for plots "
                             "(default: alongside CSV in plots/).")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        return 1
    
    return analyze_results(csv_path)


if __name__ == '__main__':
    sys.exit(main())
