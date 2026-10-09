"""
design_effect_reanalysis.py

Reanalysis of the cross-cell correlation data using the DESIGN-EFFECT
formula instead of the integrated-correlation-length (tau) formula.

Why the tau-integral model failed
---------------------------------
The cross-cell correlation profile for nested families is FLAT at
~0.70 across all Manhattan distances 1..10 (confirmed empirically).
A flat, non-decaying correlation has no finite decorrelation length,
so the integrated-correlation-length

    tau = 1 + 2 * sum_{d>=1} rho_d

grows LINEARLY with the integration window max_dist and never converges.
The predicted |beta_N| ratio tau_nn/tau_nested is therefore a deterministic
function of the arbitrary window choice (it passes through the empirical
0.362 somewhere between max_dist=1 and max_dist=2, but that is fitting the
window to the answer, not predicting it).

The correct model: design effect
--------------------------------
When L error sources share a CONSTANT pairwise correlation rho_bar (which
is what a flat correlation profile means), the effective number of
INDEPENDENT sources is given by the survey-statistics "design effect"
(a.k.a. variance inflation factor for correlated samples):

    L_eff(L, rho_bar) = L / (1 + (L - 1) * rho_bar)

  - rho_bar -> 0   : L_eff -> L          (all sources independent)
  - rho_bar -> 1   : L_eff -> 1          (all sources redundant)
  - L -> infinity  : L_eff -> 1/rho_bar  (saturates)

Trajectory success is a conjunction over the path's decision points.
|beta_N| measures how fast P(FI) drops with N, which scales with the
number of independent error sources along the path.  Hence:

    |beta_N^{nested}| / |beta_N^{non-nested}|
        approx  L_eff(L, rho_nested) / L_eff(L, rho_nonnested)

where L is the EFFECTIVE PATH LENGTH in decision points -- the single
physically-meaningful free parameter, the same for both protocols on a
given generator (the consensus path length doesn't depend on nesting).

This script
-----------
1. Loads rho_bar_nested and rho_bar_nonnested from the uploaded summaries
   (averaging over max_dist, since the profile is flat).
2. Computes the predicted |beta_N| ratio as a function of L.
3. Solves for the L that reproduces the empirical ratio 0.362 on each
   generator.
4. Checks whether maze and scatter agree on L (they should, if L is a
   geometric path-length property independent of the spatial generator).

No new data; uses only the uploaded JSON summaries.
"""

import os
import json
import math

import numpy as np


EMPIRICAL_RATIO = 0.3620689655172414
UPLOADS = '/mnt/user-data/uploads'
UPLOADS = './cross_cell_correlation_test'


def design_effect_Leff(L, rho_bar):
    """Effective independent source count under constant pairwise corr."""
    rho_bar = max(rho_bar, 0.0)   # negative corr clipped to 0 (independent)
    denom = 1.0 + (L - 1) * rho_bar
    return L / denom if denom > 0 else float('inf')


def predicted_ratio(L, rho_nested, rho_nonnested):
    """|beta_N^nested| / |beta_N^nonnested| via design-effect L_eff ratio."""
    Leff_n  = design_effect_Leff(L, rho_nested)
    Leff_nn = design_effect_Leff(L, rho_nonnested)
    return Leff_n / Leff_nn


def solve_L_for_ratio(target_ratio, rho_nested, rho_nonnested,
                     L_lo=1.0, L_hi=200.0, tol=1e-6):
    """Bisection for the L that gives predicted_ratio == target_ratio.

    predicted_ratio is monotone decreasing in L (more decision points =>
    nested saturation bites harder => smaller ratio), so bisection is safe.
    """
    f = lambda L: predicted_ratio(L, rho_nested, rho_nonnested) - target_ratio
    f_lo, f_hi = f(L_lo), f(L_hi)
    if f_lo * f_hi > 0:
        return None  # no root in range
    for _ in range(200):
        mid = 0.5 * (L_lo + L_hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            L_hi = mid
        else:
            L_lo, f_lo = mid, f_mid
    return 0.5 * (L_lo + L_hi)


def load_rho_bars():
    """Load and average rho_bar over all max_dist values per generator."""
    rho_nested = {}
    rho_nonnested = {}
    for gen in ['maze', 'scatter']:
        rns, rnns = [], []
        for md in [1, 2, 3, 6, 10]:
            p = os.path.join(UPLOADS, f"{gen}/{gen}_md={md}_cross_cell_summary.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                s = json.load(f)
            rns.append(np.mean(list(s['profile_nested'].values())))
            rnns.append(np.mean(list(s['profile_non_nested'].values())))
        if rns:
            rho_nested[gen] = float(np.mean(rns))
            rho_nonnested[gen] = float(np.mean(rnns))
    return rho_nested, rho_nonnested


def main():
    rho_nested, rho_nonnested = load_rho_bars()

    print("=" * 64)
    print("DESIGN-EFFECT REANALYSIS")
    print("=" * 64)
    print()
    print("Flat cross-cell correlation (averaged over max_dist):")
    for gen in ['maze', 'scatter']:
        print(f"  {gen:>8}: rho_nested = {rho_nested[gen]:.4f}, "
              f"rho_non_nested = {rho_nonnested[gen]:.4f}")
    print()
    print(f"Empirical |beta_N| ratio to reproduce: {EMPIRICAL_RATIO:.4f}")
    print(f"  (= |beta_N^nested| / |beta_N^non-nested| = 0.21 / 0.58)")
    print()

    # --- Solve for effective path length L on each generator ---
    print("-" * 64)
    print("Solving for effective path length L (the single free parameter)")
    print("-" * 64)
    L_solutions = {}
    for gen in ['maze', 'scatter']:
        rn, rnn = rho_nested[gen], rho_nonnested[gen]
        L_star = solve_L_for_ratio(EMPIRICAL_RATIO, rn, rnn)
        L_solutions[gen] = L_star
        if L_star is not None:
            Leff_n  = design_effect_Leff(L_star, rn)
            Leff_nn = design_effect_Leff(L_star, rnn)
            print(f"  {gen:>8}: L* = {L_star:.2f} decision points")
            print(f"           L_eff(nested)     = {Leff_n:.2f}  "
                  f"(saturates at 1/rho = {1.0/max(rn,1e-9):.2f})")
            print(f"           L_eff(non-nested) = {Leff_nn:.2f}  "
                  f"(approx L, since rho~0)")
        else:
            print(f"  {gen:>8}: no L reproduces the empirical ratio in range")
    print()

    # --- Agreement check ---
    print("-" * 64)
    print("Cross-generator agreement on L")
    print("-" * 64)
    if all(L_solutions.get(g) for g in ['maze', 'scatter']):
        Lm, Ls = L_solutions['maze'], L_solutions['scatter']
        print(f"  L*(maze)    = {Lm:.2f}")
        print(f"  L*(scatter) = {Ls:.2f}")
        print(f"  difference  = {abs(Lm - Ls):.2f}  "
              f"({100*abs(Lm-Ls)/((Lm+Ls)/2):.1f}% of mean)")
        if abs(Lm - Ls) / ((Lm + Ls) / 2) < 0.15:
            print("  -> AGREE within 15%: L is a generator-independent path")
            print("     property, as the mechanism predicts.  Strong support.")
        else:
            print("  -> DISAGREE: L differs across generators; either path")
            print("     lengths genuinely differ or the model is incomplete.")
    print()

    # --- Forward check: plug in a plausible fixed L and show prediction ---
    print("-" * 64)
    print("Forward prediction at fixed plausible L (sensitivity)")
    print("-" * 64)
    print(f"  {'L':>5} {'maze ratio':>12} {'scatter ratio':>14}")
    for L in [3, 4, 5, 6, 8, 10, 15]:
        rm = predicted_ratio(L, rho_nested['maze'], rho_nonnested['maze'])
        rs = predicted_ratio(L, rho_nested['scatter'], rho_nonnested['scatter'])
        marker = "  <- near empirical" if abs(rm - EMPIRICAL_RATIO) < 0.03 else ""
        print(f"  {L:>5} {rm:>12.3f} {rs:>14.3f}{marker}")
    print()

    # --- Saturation values ---
    print("-" * 64)
    print("Large-L saturation (the asymptotic prediction)")
    print("-" * 64)
    for gen in ['maze', 'scatter']:
        rn = rho_nested[gen]
        sat = (1.0 / rn) / 1.0   # L_eff_nested/L_eff_nonnested -> (1/rho)/L -> 0
        print(f"  {gen:>8}: as L->inf, L_eff(nested) -> 1/rho = "
              f"{1.0/rn:.2f}, ratio -> (1/rho)/L -> 0")
    print()
    print("  Interpretation: nested families behave like a FIXED ~1.4-1.5")
    print("  independent error sources no matter how long the path, while")
    print("  non-nested families have L_eff = L growing with path length.")
    print("  The ratio therefore falls as 1/L; matching 0.362 pins L ~ 4.")

    # --- Save summary ---
    out = {
        'empirical_ratio':     EMPIRICAL_RATIO,
        'rho_nested':          rho_nested,
        'rho_non_nested':      rho_nonnested,
        'L_solutions':         L_solutions,
        'model':               'design_effect: L_eff = L / (1 + (L-1)*rho_bar)',
    }
    out_dir = './design_effect_reanalysis/'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'design_effect_summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Summary saved to {out_dir}design_effect_summary.json")

    return out


if __name__ == '__main__':
    main()
