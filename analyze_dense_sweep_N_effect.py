"""
analyze_dense_sweep_N_effect.py

Post-run analysis for the dense F*-sweep on 60x30 with N-varied cells.
Addresses the two open questions from the n=30 60x30 result:

  (1) Convergence question: are F*_coher^(0.5) and F*_coher^(0.1) really at
      the same F*-density, or did the 0.05-wide F*-bins of the original
      analyzer mask a small graduated structure between these levels?

  (2) N-effect question: at fixed F*-density, does P(FI) depend on N
      (number of environments)?  The 60x30 result at n=30 showed a hint
      of N-effect at 2-rho bins, but n=30 + small dynamic range in N
      couldn't resolve it.

Usage
-----
    from analyze_dense_sweep_N_effect import analyze_dense_sweep
    summary = analyze_dense_sweep(
        sweep_df,
        out_dir='./logs/dense_sweep_analysis/',
    )

Produces three plots and a printed summary:
  - plot_pfi_vs_F_by_N.png    : P(FI) vs F*_realized, coloured by N
  - plot_pfi_vs_F_by_rho.png  : P(FI) vs F*_realized, coloured by rho (for comparison)
  - plot_logistic_fit.png     : Logistic regression fit with N-stratified curves
"""

import os
import math
import warnings

import numpy as np
import pandas as pd

# Optional dependency: statsmodels for proper inference
try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ==============================================================
# Plot 1: P(FI) vs F*_realized, coloured by N
# ==============================================================

def plot_pfi_vs_F_by_N(sweep_df, out_path, n_color_levels=None):
    """
    Scatter plot of P(FI) vs realized F*-density, with marker colour
    indicating N (number of environments in the family).  Each point is
    a (rho, N) cell with its mean P(FI) over n_seeds.

    If the N-effect is real, points with low N (e.g. N=2) will sit
    visibly above points with high N at matched F*-density.  If P(FI)
    depends only on F*, the points will fall on a single curve
    regardless of N.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Must have columns: rho, N, F_star_density, C_interp.
    out_path : str
        Where to save the plot PNG.
    n_color_levels : list[int], optional
        Restrict to these N values for clarity.  Default: all N present.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = sweep_df.copy()
    df['fi_holds'] = df['C_interp'].isin(['marginal', 'free'])

    # Per-cell aggregation (one point per (rho, N) cell)
    cells = (df.groupby(['rho', 'N'])
               .agg(p_fi=('fi_holds', 'mean'),
                    F_real=('F_star_density', 'mean'),
                    F_std=('F_star_density', 'std'),
                    n=('fi_holds', 'count'))
               .reset_index())
    # Binomial SE per cell
    cells['se'] = np.sqrt(cells['p_fi'] * (1 - cells['p_fi']) / cells['n'])

    if n_color_levels is None:
        n_color_levels = sorted(cells['N'].unique())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(n_color_levels)))
    n_to_color = dict(zip(n_color_levels, cmap))

    for N_val in n_color_levels:
        sub = cells[cells['N'] == N_val]
        if len(sub) == 0:
            continue
        ax.errorbar(
            sub['F_real'], sub['p_fi'],
            xerr=sub['F_std'], yerr=sub['se'],
            fmt='o', color=n_to_color[N_val],
            label=f'N = {int(N_val)}  ({len(sub)} cells)',
            capsize=3, markersize=8, alpha=0.85, linewidth=1.5,
        )

    # Reference horizontal lines at the FI-level thresholds
    for level, label in [(0.9, 'P(FI)=0.9 (onset)'),
                         (0.5, 'P(FI)=0.5 (median)'),
                         (0.1, 'P(FI)=0.1 (deep)')]:
        ax.axhline(level, color='gray', linestyle=':', alpha=0.4)
        ax.annotate(label, (1.0, level), xytext=(-2, 2),
                    textcoords='offset points',
                    ha='right', fontsize=7, color='gray')

    # Nested-bridge reference
    ax.axvline(0.699, color='green', linestyle='--', alpha=0.5,
               label=r'nested $\phi_c^{nest}=0.699$')

    ax.set_xlabel(r'realized $|F^*| / |\mathcal{S}|$', fontsize=11)
    ax.set_ylabel('P(FI holds)', fontsize=11)
    ax.set_title(
        'P(FI) vs F*-density, coloured by N  '
        '(overlap across N = geometry sufficient; '
        'separation = N-effect real)'
    )
    ax.set_xlim(0.65, 0.95)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ==============================================================
# Plot 1b: P(FI) vs F*_realized, coloured by rho (comparison)
# ==============================================================

def plot_pfi_vs_F_by_rho(sweep_df, out_path):
    """
    Same as plot_pfi_vs_F_by_N but coloured by rho.  Useful for
    visual comparison: in the N-coloured plot, overlap means geometry
    suffices; in the rho-coloured plot, the same overlap means rho
    doesn't matter at fixed F*.  Both should overlap if F*-density is
    the order parameter.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = sweep_df.copy()
    df['fi_holds'] = df['C_interp'].isin(['marginal', 'free'])
    cells = (df.groupby(['rho', 'N'])
               .agg(p_fi=('fi_holds', 'mean'),
                    F_real=('F_star_density', 'mean'),
                    F_std=('F_star_density', 'std'),
                    n=('fi_holds', 'count'))
               .reset_index())
    cells['se'] = np.sqrt(cells['p_fi'] * (1 - cells['p_fi']) / cells['n'])

    rhos = sorted(cells['rho'].unique())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.plasma(np.linspace(0.10, 0.85, len(rhos)))
    rho_to_color = dict(zip(rhos, cmap))

    for rho_val in rhos:
        sub = cells[cells['rho'] == rho_val]
        ax.errorbar(
            sub['F_real'], sub['p_fi'],
            xerr=sub['F_std'], yerr=sub['se'],
            fmt='o', color=rho_to_color[rho_val],
            label=f'rho = {rho_val:.2f}  ({len(sub)} cells)',
            capsize=3, markersize=8, alpha=0.85, linewidth=1.5,
        )

    for level in [0.9, 0.5, 0.1]:
        ax.axhline(level, color='gray', linestyle=':', alpha=0.4)

    ax.axvline(0.699, color='green', linestyle='--', alpha=0.5,
               label=r'nested $\phi_c^{nest}$')
    ax.set_xlabel(r'realized $|F^*| / |\mathcal{S}|$', fontsize=11)
    ax.set_ylabel('P(FI holds)', fontsize=11)
    ax.set_title('P(FI) vs F*-density, coloured by rho')
    ax.set_xlim(0.65, 0.95)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ==============================================================
# Plot 2 + regression: logistic fit of P(FI) ~ F* + N
# ==============================================================

def logistic_fit_pfi(sweep_df, out_plot_path=None):
    """
    Fit a logistic regression P(FI = 1) ~ logit(alpha + beta_F * F* + beta_N * N)
    on the seed-level data (one row = one (env_family, seed) outcome).

    Returns
    -------
    dict with:
      - 'coefs'      : DataFrame of (estimate, se, z, p) for alpha, F*, N
      - 'F_slope'    : (beta_F estimate, se, p)  -- transition steepness
      - 'N_effect'   : (beta_N estimate, se, p)  -- N-effect at fixed F*
      - 'width_F'    : transition width in F*, estimated as 1 / |beta_F|
      - 'fit_kind'   : 'statsmodels' or 'numpy_fallback'

    Also produces a plot at `out_plot_path` (if provided) showing the
    data with the fitted curves at each N value, plus a comparison
    curve where N is dropped.
    """
    df = sweep_df.copy()
    df['fi_holds'] = df['C_interp'].isin(['marginal', 'free']).astype(int)

    # The regression operates on seed-level data, not cell aggregates --
    # this gives correct per-observation weighting for the logistic.
    X_cols = ['F_star_density', 'N']
    y = df['fi_holds'].values
    X = df[X_cols].copy()

    result = {}

    if _HAS_STATSMODELS:
        X_with_const = sm.add_constant(X.values)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # MLE for logistic; method='bfgs' helps when there's quasi-separation
            try:
                model = sm.Logit(y, X_with_const).fit(disp=0, method='newton')
            except Exception:
                model = sm.Logit(y, X_with_const).fit(disp=0, method='bfgs')

        coef_names = ['intercept', 'F_star', 'N']
        coefs_df = pd.DataFrame({
            'name':     coef_names,
            'estimate': model.params,
            'se':       model.bse,
            'z':        model.tvalues,
            'p':        model.pvalues,
        })
        result['coefs']    = coefs_df
        result['F_slope']  = {
            'estimate': float(model.params[1]),
            'se':       float(model.bse[1]),
            'p':        float(model.pvalues[1]),
        }
        result['N_effect'] = {
            'estimate': float(model.params[2]),
            'se':       float(model.bse[2]),
            'p':        float(model.pvalues[2]),
        }
        result['fit_kind']    = 'statsmodels'
        result['log_likelihood'] = float(model.llf)
        result['model']    = model

    else:
        # Numpy fallback: simple coordinate-descent IRLS for logistic
        # Not as robust as statsmodels but works in a pinch
        beta = _logistic_irls_fallback(X.values, y)
        # Approximate SEs via the Hessian at the fitted point
        ses = _logistic_se_fallback(X.values, y, beta)
        z = beta / ses
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(np.abs(z)))
        coef_names = ['intercept', 'F_star', 'N']
        coefs_df = pd.DataFrame({
            'name':     coef_names,
            'estimate': beta,
            'se':       ses,
            'z':        z,
            'p':        p,
        })
        result['coefs']    = coefs_df
        result['F_slope']  = {'estimate': beta[1], 'se': ses[1], 'p': p[1]}
        result['N_effect'] = {'estimate': beta[2], 'se': ses[2], 'p': p[2]}
        result['fit_kind'] = 'numpy_fallback'

    result['width_F'] = abs(1.0 / result['F_slope']['estimate']) \
                          if result['F_slope']['estimate'] != 0 else float('inf')

    if out_plot_path is not None:
        _plot_logistic_fit(df, result, out_plot_path)

    return result


def _logistic_irls_fallback(X, y, max_iter=50, tol=1e-6):
    """Simple iteratively-reweighted-least-squares logistic regression."""
    X_with_const = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X_with_const.shape[1])
    for _ in range(max_iter):
        eta = X_with_const @ beta
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p) + 1e-10
        z_target = eta + (y - p) / W
        WX = X_with_const * W[:, None]
        try:
            beta_new = np.linalg.solve(X_with_const.T @ WX,
                                       X_with_const.T @ (W * z_target))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _logistic_se_fallback(X, y, beta):
    """Approximate SEs via inverse Fisher information."""
    X_with_const = np.column_stack([np.ones(len(X)), X])
    eta = np.clip(X_with_const @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    W = p * (1 - p) + 1e-10
    fisher = X_with_const.T @ (X_with_const * W[:, None])
    try:
        cov = np.linalg.inv(fisher)
        return np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return np.full(X_with_const.shape[1], np.nan)


def _plot_logistic_fit(df, fit_result, out_path):
    """Plot the fitted curves vs the data, stratified by N."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = df.copy()
    df['fi_holds'] = df['C_interp'].isin(['marginal', 'free']).astype(int)
    cells = (df.groupby(['rho', 'N'])
               .agg(p_fi=('fi_holds', 'mean'),
                    F_real=('F_star_density', 'mean'),
                    n=('fi_holds', 'count'))
               .reset_index())
    cells['se'] = np.sqrt(cells['p_fi'] * (1 - cells['p_fi']) / cells['n'])

    # Plot data
    Ns = sorted(cells['N'].unique())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ns)))
    n_to_color = dict(zip(Ns, cmap))

    for N_val in Ns:
        sub = cells[cells['N'] == N_val]
        ax.errorbar(
            sub['F_real'], sub['p_fi'], yerr=sub['se'],
            fmt='o', color=n_to_color[N_val],
            label=f'N = {int(N_val)}  (data)',
            capsize=3, markersize=8, alpha=0.85,
        )

    # Plot fitted curves at each N value
    beta = fit_result['coefs']['estimate'].values
    F_grid = np.linspace(cells['F_real'].min() - 0.02,
                         cells['F_real'].max() + 0.02, 200)
    for N_val in Ns:
        eta = beta[0] + beta[1] * F_grid + beta[2] * N_val
        eta = np.clip(eta, -30, 30)
        p_curve = 1.0 / (1.0 + np.exp(-eta))
        ax.plot(F_grid, p_curve, '--',
                color=n_to_color[N_val], alpha=0.6, linewidth=1.5,
                label=f'N = {int(N_val)}  (fit)')

    # Annotation with key fit results
    F_eff_text = (
        f"F* slope: {fit_result['F_slope']['estimate']:.2f} "
        f"(SE {fit_result['F_slope']['se']:.2f}, "
        f"p = {fit_result['F_slope']['p']:.4f})\n"
        f"N effect: {fit_result['N_effect']['estimate']:.3f} "
        f"(SE {fit_result['N_effect']['se']:.3f}, "
        f"p = {fit_result['N_effect']['p']:.4f})\n"
        f"Transition width 1/|F*-slope|: {fit_result['width_F']:.3f}"
    )
    ax.text(0.02, 0.02, F_eff_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    for level in [0.9, 0.5, 0.1]:
        ax.axhline(level, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(0.699, color='green', linestyle='--', alpha=0.5,
               label=r'nested $\phi_c^{nest}$')

    ax.set_xlabel(r'realized $|F^*| / |\mathcal{S}|$', fontsize=11)
    ax.set_ylabel('P(FI holds)', fontsize=11)
    ax.set_title(
        f'Logistic fit P(FI) ~ F* + N  '
        f'(N effect p = {fit_result["N_effect"]["p"]:.4f})'
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ==============================================================
# Driver
# ==============================================================

def analyze_dense_sweep(sweep_df, out_dir='./dense_sweep_analysis/',
                       n_color_levels=None):
    """
    Run all three analyses and print a summary.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Sweep output with columns: rho, N, F_star_density, C_interp, etc.
    out_dir : str
        Directory for plot output.
    n_color_levels : list[int], optional
        Override the N-values to display in plots.

    Returns
    -------
    dict with keys: 'fit', 'plots' (list of paths)
    """
    os.makedirs(out_dir, exist_ok=True)
    plots = []

    p1 = os.path.join(out_dir, 'plot_pfi_vs_F_by_N.png')
    plot_pfi_vs_F_by_N(sweep_df, p1, n_color_levels=n_color_levels)
    plots.append(p1)

    p2 = os.path.join(out_dir, 'plot_pfi_vs_F_by_rho.png')
    plot_pfi_vs_F_by_rho(sweep_df, p2)
    plots.append(p2)

    p3 = os.path.join(out_dir, 'plot_logistic_fit.png')
    fit = logistic_fit_pfi(sweep_df, out_plot_path=p3)
    plots.append(p3)

    # ----- Print summary -----
    print()
    print('=' * 70)
    print('DENSE F*-SWEEP ANALYSIS: N-effect and transition shape')
    print('=' * 70)
    print(f"Fit kind: {fit['fit_kind']}")
    print()
    print("Logistic regression: P(FI) ~ logit(alpha + beta_F * F* + beta_N * N)")
    print()
    print(fit['coefs'].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print()
    print('-- Key inferences --')
    # Transition-width interpretation: full 10%-90% logistic transition
    # spans approximately 4.4 / |beta_F| in F*-density (from the logit-link).
    transition_10_90 = 4.4 / abs(fit['F_slope']['estimate']) \
                          if fit['F_slope']['estimate'] != 0 else float('inf')
    shape_label = ('SHARP (10%-90% width < 0.10)' if transition_10_90 < 0.10
                   else 'NARROW (0.10 <= width < 0.20)' if transition_10_90 < 0.20
                   else 'GRADUATED (width >= 0.20)')
    print(f"F*-density slope (beta_F): {fit['F_slope']['estimate']:+.3f}  "
          f"(SE {fit['F_slope']['se']:.3f}, p = {fit['F_slope']['p']:.4f})")
    print(f"  -> 10%-90% transition width: {transition_10_90:.3f} in F*-density "
          f"({shape_label})")
    print()
    print(f"N effect (beta_N): {fit['N_effect']['estimate']:+.4f}  "
          f"(SE {fit['N_effect']['se']:.4f}, p = {fit['N_effect']['p']:.4f})")
    print("  ")
    print("  Voting-argument prediction at fixed F*-density:")
    print("    higher N -> lower per-env rho (to maintain F*)")
    print("    lower rho -> wider per-cell margins")
    print("    -> higher P(FI), i.e. POSITIVE beta_N")
    print("  ")
    if fit['N_effect']['p'] < 0.05:
        if fit['N_effect']['estimate'] > 0:
            direction = ('POSITIVE: P(FI) rises with N at fixed F*.  '
                         'CONSISTENT with the voting-argument prediction; '
                         'voting margin is operative beyond geometry.')
        else:
            direction = ('NEGATIVE: P(FI) falls with N at fixed F*.  '
                         'OPPOSITE to the voting-argument prediction; '
                         'more environments produce more value-mixture '
                         'disagreement, lowering P(FI) even with looser '
                         'individual margins.')
        print(f"  -> N-effect is STATISTICALLY RESOLVED at p < 0.05: {direction}")

        # Effect size at the inflection point (p ~ 0.5): a one-unit change
        # in N shifts logit(p) by beta_N, which shifts p by ~0.25 * beta_N
        # at p = 0.5 (where dp/d(logit) = p(1-p) is maximal).
        delta_N = 3   # from N=2 to N=5
        delta_p = abs(0.25 * fit['N_effect']['estimate'] * delta_N)
        print(f"  -> Effect size: at the midpoint of the transition, "
              f"N=2->5 shifts P(FI) by ~{delta_p:.3f}")
    else:
        print(f"  -> N-effect is NOT statistically resolvable at p < 0.05; "
              f"data consistent with F*-density being the sole order parameter "
              f"(possible effect direction noted in beta_N sign above).")
    print()
    print('-- Plots written to --')
    for p in plots:
        print(f"  {p}")
    print('=' * 70)

    return {'fit': fit, 'plots': plots}
