"""
phase1_corrected_protocol_geometry_check.py

Phase 1: pure-geometry sanity check for the CORRECTED non-nested protocol.

Background
----------
The previous build_family_for_test() implemented non-nested ('iid') mode
as "N envs all at the same density rho", which does not match the
original framework definition.  The corrected definition is:

  Non-nested (iid):
    - density schedule: rho_i = (i+1)/N * rho_max   (graduated, linear)
    - per-env seeds:    independent skeleton seed AND independent
                        obstacle seed for each env
    - obstacle structure: each env's obstacles are sampled independently
                          at its assigned density

  Nested:
    - density schedule: rho_i = (i+1)/N * rho_max   (same as non-nested)
    - per-env seeds:    SHARED skeleton seed AND SHARED obstacle-
                        permutation seed across all N envs
    - obstacle structure: each env's obstacles are a prefix of the
                          shared shuffle at length proportional to rho_i;
                          so O_nu_1 subset O_nu_2 subset ... subset O_nu_N

Both protocols share the same per-env density distribution.  What
differs is the JOINT structure across envs:
  - nested has positively correlated occlusion events (shared skeleton +
    nested obstacle prefixes)
  - non-nested has independent occlusion events (independent skeletons,
    independent obstacle samples per env)

This script verifies three things:

  (A) BUILDER CORRECTNESS: per-env density matches schedule;
      nested has subset-inclusion; non-nested has independent samples.

  (B) F*-DENSITY SENSIBILITY: realized phi values look reasonable on
      both protocols across many master seeds.

  (C) F* approx F_nu_N CLAIM: in both protocols, F* is approximately
      determined by the densest env, so the two protocols should put
      F* at roughly the same place at matched (rho_max, N).

No Q-tables.  Runs in minutes.
"""

import os
import json
import math

import numpy as np
import pandas as pd
from tqdm import tqdm


# ==============================================================
# CORRECTED build_family_for_test
# ==============================================================
#
# IMPORTANT: this is the patch the user should propagate to the real
# build_family_for_test.  Both modes now use the graduated density
# schedule rho_i = (i+1)/N * rho_max.  Non-nested has independent
# seeds throughout; nested has shared skeleton AND shared obstacle
# permutation seed.

def build_family_corrected(
    rho, N, mode, master_seed, grid_w, grid_h,
    lambda_cliff=100.0, grid_family='maze',
    nested_schedule='linear', full_env_name=True,
):
    """
    Build a family of N grid envs under the CORRECTED protocols.

    rho here is interpreted as rho_max (the densest env's density)
    in BOTH modes.  The density schedule is graduated:
        rho_i = (i+1)/N * rho_max   (linear schedule)

    Modes:
      'iid' / 'non_nested' : independent skeleton seed AND independent
                             obstacle seed per env.  Obstacles at each
                             density are independently sampled.
      'shared_skel'        : shared skeleton seed, Obstacles at each
                             density are independently sampled.
      'nested'             : shared skeleton seed, shared obstacle-
                             permutation seed.  Obstacles at each density
                             are a strict prefix of the shared shuffle.
    """
    from GridWorldCanonicalBenchmarkSuite import (
        RandomMazeFamily, RandomScatterFamily, SizeSpec,
    )

    if grid_family == 'maze':
        fam = RandomMazeFamily(C=10.0, lambda_cliff=lambda_cliff)
    elif grid_family == 'scatter':
        fam = RandomScatterFamily(C=10.0, lambda_cliff=lambda_cliff,
                                  connectivity_mode='carve')
    else:
        raise ValueError(f"unknown grid_family {grid_family!r}")

    size_spec = SizeSpec(grid_w, grid_h)

    # Density schedule (shared by both modes)
    if nested_schedule == 'linear':
        rho_schedule = [round(((i + 1) / N) * rho, 4) for i in range(N)]
    elif nested_schedule == 'equal':
        rho_schedule = [rho] * N
    else:
        raise ValueError(f"unknown nested_schedule {nested_schedule!r}")

    envs = []

    if mode in ('iid', 'non_nested'):
        for i, rho_i in enumerate(rho_schedule):
            # Independent skeleton AND obstacle seed per env
            seed_i = hash((master_seed, 'skel', i)) & 0x7FFFFFFF
            obs_i  = hash((master_seed, 'obs',  i)) & 0x7FFFFFFF
            env = fam.make(
                size=size_spec, scale=1, walls_perc=rho_i,
                seed=seed_i, obstacle_seed=obs_i, verbose=False,
            )
            if full_env_name:
                env.name = (f'{env.name}-non_nested-ms{master_seed}-'
                        f'i{i}-rho{rho_i:.3f}')
            envs.append(env)

    elif mode == 'shared_skel':
        # SHARED skeleton seed 
        shared_skel = master_seed & 0x7FFFFFFF
        for i, rho_i in enumerate(rho_schedule):
            # Independent obstacle seed per env
            obs_i  = hash((master_seed, 'obs',  i)) & 0x7FFFFFFF
            env = fam.make(
                size=size_spec, scale=1, walls_perc=rho_i,
                seed=shared_skel, obstacle_seed=obs_i, verbose=False,
            )
            if full_env_name:
                env.name = (f'{env.name}-shared_skel-ms{master_seed}-'
                        f'rhomax{rho:.3f}-N{N}-'
                        f'i{i}-rho{rho_i:.3f}')
            envs.append(env)

    elif mode == 'nested':
        # SHARED skeleton seed and SHARED obstacle-permutation seed
        shared_skel = master_seed & 0x7FFFFFFF
        shared_obs  = hash((master_seed, 'nested_obs')) & 0x7FFFFFFF
        for i, rho_i in enumerate(rho_schedule):
            env = fam.make(
                size=size_spec, scale=1, walls_perc=rho_i,
                seed=shared_skel, obstacle_seed=shared_obs, verbose=False,
            )
            if full_env_name:
                env.name = (f'{env.name}-nested-{nested_schedule}-'
                        f'rhomax{rho:.3f}-N{N}-ms{master_seed}-'
                        f'i{i}-rho{rho_i:.3f}')
            envs.append(env)

    else:
        raise ValueError(f"unknown mode {mode!r}")

    return envs, rho_schedule


# ==============================================================
# Geometric utilities
# ==============================================================

def compute_f_star(envs, grid_w, grid_h):
    """Realized F*-density: fraction of cells free in EVERY env."""
    free_mask = np.ones((grid_w, grid_h), dtype=bool)
    for env in envs:
        env_walls = np.zeros((grid_w, grid_h), dtype=bool)
        for (x, y) in env.walls:
            if 0 <= x < grid_w and 0 <= y < grid_h:
                env_walls[x, y] = True
        free_mask &= ~env_walls
    return float(free_mask.sum() / (grid_w * grid_h)), free_mask


def per_env_density(env, grid_w, grid_h):
    """Fraction of grid cells that are walls in this env."""
    n_walls = sum(1 for (x, y) in env.walls
                  if 0 <= x < grid_w and 0 <= y < grid_h)
    return n_walls / (grid_w * grid_h)


def check_nested_inclusion(envs):
    """
    For nested protocol, verify O_nu_1 subset O_nu_2 subset ... O_nu_N.
    Returns list of booleans, one per consecutive pair.
    """
    inclusions = []
    for i in range(1, len(envs)):
        o_prev = set(envs[i - 1].walls)
        o_curr = set(envs[i].walls)
        inclusions.append(o_prev.issubset(o_curr))
    return inclusions


def cross_env_overlap(envs):
    """
    For non-nested protocol, measure how much obstacle sets overlap
    across envs.  Returns the average Jaccard similarity over all
    pairs (i, j) with i < j.

    For genuinely independent samples, this should be small (proportional
    to the product of individual densities).  For accidental correlation,
    it would be high.
    """
    pairs = []
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            o_i = set(envs[i].walls)
            o_j = set(envs[j].walls)
            if not o_i and not o_j:
                pairs.append(0.0)
            else:
                inter = len(o_i & o_j)
                union = len(o_i | o_j)
                pairs.append(inter / union if union else 0.0)
    return float(np.mean(pairs)) if pairs else 0.0


# ==============================================================
# Test A: builder correctness
# ==============================================================

def test_builder_correctness(grid_w=60, grid_h=30, n_check_seeds=10,
                              grid_family='maze'):
    """
    Build small families under both modes and verify:
      - per-env densities match the schedule rho_i = (i+1)/N * rho_max
      - nested has O_nu_1 subset ... subset O_nu_N
      - non-nested envs have low cross-env obstacle overlap
        (consistent with independent samples)
    """
    print("=" * 64)
    print("TEST A: builder correctness")
    print("=" * 64)
    print()

    test_params = [
        (0.50, 5),
        (0.70, 5),
        (1.00, 5),
        (0.50, 10),
        (0.70, 10),
        (1.00, 10),
    ]

    for (rho_max, N) in test_params:
        expected_schedule = [round(((i + 1) / N) * rho_max, 4)
                             for i in range(N)]
        print(f"  (rho_max={rho_max}, N={N}), expected schedule:")
        print(f"    {expected_schedule}")
        print()

        # --- Non-nested ---
        per_env_dens_nn = [[] for _ in range(N)]
        cross_overlaps  = []
        for seed in range(n_check_seeds):
            envs, _ = build_family_corrected(
                rho=rho_max, N=N, mode='non_nested',
                master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                grid_family=grid_family,
            )
            for i, env in enumerate(envs):
                per_env_dens_nn[i].append(
                    per_env_density(env, grid_w, grid_h)
                )
            cross_overlaps.append(cross_env_overlap(envs))

        print(f"  Non-nested:")
        print(f"    {'i':>3} {'rho_i target':>14} {'realized mean':>14} "
              f"{'(SD)':>8}")
        for i in range(N):
            mean = np.mean(per_env_dens_nn[i])
            sd   = np.std(per_env_dens_nn[i])
            print(f"    {i:>3} {expected_schedule[i]:>14.4f} "
                  f"{mean:>14.4f} {sd:>8.4f}")
        print(f"    avg cross-env obstacle Jaccard (should be small): "
              f"{np.mean(cross_overlaps):.4f} +/- "
              f"{np.std(cross_overlaps):.4f}")
        print()


        # --- Shared-Skeleton ---
        per_env_dens_sk = [[] for _ in range(N)]
        cross_overlaps  = []
        for seed in range(n_check_seeds):
            envs, _ = build_family_corrected(
                rho=rho_max, N=N, mode='shared_skel',
                master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                grid_family=grid_family,
            )
            for i, env in enumerate(envs):
                per_env_dens_sk[i].append(
                    per_env_density(env, grid_w, grid_h)
                )
            cross_overlaps.append(cross_env_overlap(envs))

        print(f"  Shared-Skeleton:")
        print(f"    {'i':>3} {'rho_i target':>14} {'realized mean':>14} "
              f"{'(SD)':>8}")
        for i in range(N):
            mean = np.mean(per_env_dens_sk[i])
            sd   = np.std(per_env_dens_sk[i])
            print(f"    {i:>3} {expected_schedule[i]:>14.4f} "
                  f"{mean:>14.4f} {sd:>8.4f}")
        print(f"    avg cross-env obstacle Jaccard (should be small): "
              f"{np.mean(cross_overlaps):.4f} +/- "
              f"{np.std(cross_overlaps):.4f}")
        print()

        # --- Nested ---
        per_env_dens_n = [[] for _ in range(N)]
        inclusion_fracs = []
        for seed in range(n_check_seeds):
            envs, _ = build_family_corrected(
                rho=rho_max, N=N, mode='nested',
                master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                grid_family=grid_family,
            )
            for i, env in enumerate(envs):
                per_env_dens_n[i].append(
                    per_env_density(env, grid_w, grid_h)
                )
            incl = check_nested_inclusion(envs)
            inclusion_fracs.append(sum(incl) / len(incl) if incl else 1.0)

        print(f"  Nested:")
        print(f"    {'i':>3} {'rho_i target':>14} {'realized mean':>14} "
              f"{'(SD)':>8}")
        for i in range(N):
            mean = np.mean(per_env_dens_n[i])
            sd   = np.std(per_env_dens_n[i])
            print(f"    {i:>3} {expected_schedule[i]:>14.4f} "
                  f"{mean:>14.4f} {sd:>8.4f}")
        # Inclusion fraction should be 1.0 if the maze generator has the
        # monotonic-prefix property when seeds are fixed and walls_perc
        # rises.  If not, build_family_corrected may need to fall back
        # to the manual prefix construction (separate concern).
        print(f"    fraction of consecutive pairs with "
              f"O_nu_i subset O_nu_(i+1): "
              f"{np.mean(inclusion_fracs):.3f}")
        print()


# ==============================================================
# Test B: F*-density sensibility
# ==============================================================

def test_f_star_sensibility(grid_w=60, grid_h=30, n_seeds=50,
                             grid_family='maze'):
    """
    For both protocols at matched (rho_max, N), compute the realized
    F*-density across master_seeds.  Reports the mean, SD, and the
    nominal-iid prediction phi_iid = (1-rho_max)^N (which would apply
    if all envs were independently at density rho_max -- which is NOT
    what the corrected non-nested does).
    """
    print("=" * 64)
    print("TEST B: F*-density sensibility")
    print("=" * 64)
    print()

    cells = [
        (0.50, 2), (0.50, 5), (0.50, 10),
        (0.60, 2), (0.60, 5), (0.60, 10),
        (0.70, 2), (0.70, 5), (0.70, 10),
        (0.80, 2), (0.80, 5), (0.80, 10),
        (0.90, 2), (0.90, 5), (0.90, 10),
        (1.00, 2), (1.00, 5), (1.00, 10),
    ]
    rows = []
    print(f"  {'rho_max':>8} {'N':>3} "
          f"{'phi^non_nested':>16} {'(SD)':>8} "
          f"{'phi^shared_skel':>16} {'(SD)':>8} "
          f"{'phi^nested':>12} {'(SD)':>8}")
    for (rho_max, N) in tqdm(cells, desc='F* sensibility'):
        phi_nn = []
        phi_sk = []
        phi_n  = []
        for seed in range(n_seeds):
            envs_nn, _ = build_family_corrected(
                rho=rho_max, N=N, mode='non_nested',
                master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                grid_family=grid_family,
            )
            phi_nn_seed, _ = compute_f_star(envs_nn, grid_w, grid_h)
            phi_nn.append(phi_nn_seed)

            envs_sk, _ = build_family_corrected(
                rho=rho_max, N=N, mode='shared_skel',
                master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                grid_family=grid_family,
            )
            phi_sk_seed, _ = compute_f_star(envs_sk, grid_w, grid_h)
            phi_sk.append(phi_sk_seed)

            envs_n, _ = build_family_corrected(
                rho=rho_max, N=N, mode='nested',
                master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                grid_family=grid_family,
            )
            phi_n_seed, _ = compute_f_star(envs_n, grid_w, grid_h)
            phi_n.append(phi_n_seed)

        rows.append({
            'rho_max': rho_max, 'N': N,
            'phi_non_nested':       float(np.mean(phi_nn)),
            'phi_non_nested_sd':    float(np.std(phi_nn)),
            'phi_shared_skel':       float(np.mean(phi_sk)),
            'phi_shared_skel_sd':    float(np.std(phi_sk)),
            'phi_nested':           float(np.mean(phi_n)),
            'phi_nested_sd':        float(np.std(phi_n)),
            'phi_difference (n - nn)':       float(np.mean(phi_n) - np.mean(phi_nn)),
            'phi_difference (sk - nn)':       float(np.mean(phi_sk) - np.mean(phi_nn)),
            'phi_difference (n - sk)':       float(np.mean(phi_n) - np.mean(phi_sk)),
        })
    print()
    for r in rows:
        print(f"  {r['rho_max']:>8.2f} {r['N']:>3} "
              f"{r['phi_non_nested']:>16.4f} {r['phi_non_nested_sd']:>8.4f} "
              f"{r['phi_shared_skel']:>16.4f} {r['phi_shared_skel_sd']:>8.4f} "
              f"{r['phi_nested']:>12.4f} {r['phi_nested_sd']:>8.4f}")
    print()

    return pd.DataFrame(rows)


# ==============================================================
# Test C: F* approx F_nu_N claim
# ==============================================================

def test_f_star_vs_densest_env(grid_w=60, grid_h=30, n_seeds=50,
                                grid_family='maze'):
    """
    For each protocol, measure how much F* differs from F_nu_N (the
    free set of just the densest env).  Reports:
        ratio = |F*| / |F_nu_N|     (should be close to 1 in both
                                     protocols if F_nu_N dominates)
        agreement = |F* nested| / |F* non-nested|   (should be near 1
                                                    if the three protocols
                                                    put F* in the same place)
    """
    print("=" * 64)
    print("TEST C: F* approx F_nu_N, and cross-protocol F* agreement")
    print("=" * 64)
    print()

    cells = [
        (0.50, 5), (0.60, 5), (0.70, 5), (0.80, 5), (1.0, 5),
        (0.50, 10), (0.60, 10), (0.70, 10), (1.0, 10)
    ]
    rows = []
    for (rho_max, N) in tqdm(cells, desc='F* vs F_nu_N'):
        ratios_nn = []   # |F*| / |F_nu_N| in non-nested
        ratios_sk = []   # |F*| / |F_nu_N| in shared_skel
        ratios_n  = []   # |F*| / |F_nu_N| in nested
        for seed in range(n_seeds):
            envs_nn, _ = build_family_corrected(
                rho=rho_max, N=N, mode='non_nested',
                master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                grid_family=grid_family,
            )
            phi_nn, _ = compute_f_star(envs_nn, grid_w, grid_h)
            phi_densest_nn, _ = compute_f_star([envs_nn[-1]], grid_w, grid_h)
            if phi_densest_nn > 0:
                ratios_nn.append(phi_nn / phi_densest_nn)

            envs_sk, _ = build_family_corrected(
                rho=rho_max, N=N, mode='shared_skel',
                master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                grid_family=grid_family,
            )
            phi_sk, _ = compute_f_star(envs_sk, grid_w, grid_h)
            phi_densest_sk, _ = compute_f_star([envs_sk[-1]], grid_w, grid_h)
            if phi_densest_sk > 0:
                ratios_sk.append(phi_sk / phi_densest_sk)

            envs_n, _ = build_family_corrected(
                rho=rho_max, N=N, mode='nested',
                master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                grid_family=grid_family,
            )
            phi_n, _ = compute_f_star(envs_n, grid_w, grid_h)
            phi_densest_n, _ = compute_f_star([envs_n[-1]], grid_w, grid_h)
            if phi_densest_n > 0:
                ratios_n.append(phi_n / phi_densest_n)

        rows.append({
            'rho_max': rho_max, 'N': N,
            'phi_over_phi_densest_non_nested':
                float(np.mean(ratios_nn)) if ratios_nn else float('nan'),
            'phi_over_phi_densest_non_nested_sd':
                float(np.std(ratios_nn)) if ratios_nn else float('nan'),
            'phi_over_phi_densest_shared_skel':
                float(np.mean(ratios_sk)) if ratios_sk else float('nan'),
            'phi_over_phi_densest_shared_skel_sd':
                float(np.std(ratios_sk)) if ratios_sk else float('nan'),
            'phi_over_phi_densest_nested':
                float(np.mean(ratios_n)) if ratios_n else float('nan'),
            'phi_over_phi_densest_nested_sd':
                float(np.std(ratios_n)) if ratios_n else float('nan'),
        })

    print(f"  {'rho_max':>8} {'N':>3} "
          f"{'F*/F_nu_N (nn)':>16} {'(SD)':>8} "
          f"{'F*/F_nu_N (sk)':>16} {'(SD)':>8} "
          f"{'F*/F_nu_N (n)':>16} {'(SD)':>8}")
    for r in rows:
        print(f"  {r['rho_max']:>8.2f} {r['N']:>3} "
              f"{r['phi_over_phi_densest_non_nested']:>16.4f} "
              f"{r['phi_over_phi_densest_non_nested_sd']:>8.4f} "
              f"{r['phi_over_phi_densest_shared_skel']:>16.4f} "
              f"{r['phi_over_phi_densest_shared_skel_sd']:>8.4f} "
              f"{r['phi_over_phi_densest_nested']:>16.4f} "
              f"{r['phi_over_phi_densest_nested_sd']:>8.4f}")
    print()
    print("  Interpretation:")
    print("    Ratio close to 1.0 -> F* is essentially F_nu_N (densest env)")
    print("    Ratio < 1.0       -> lower-density envs DO further shrink F*")
    print("    Nested ratio is 1.0 by construction (F* = F_nu_N exactly).")
    print("    Shared Skeleton ratio is 1.0 by construction (F* = F_nu_N exactly).")    
    print("    Non-nested ratio < 1 quantifies how much lower-density envs")
    print("    independently contribute additional obstacles to the")
    print("    consensus.")
    print()

    return pd.DataFrame(rows)


# ==============================================================
# Main driver
# ==============================================================

def main(out_dir='./phase1_geometry_check/',
         grid_w=60, grid_h=30,
         n_check_seeds=10, n_seeds=50,
         grid_family='maze'):
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("##" * 32)
    print("# Phase 1: corrected-protocol geometry check")
    print("##" * 32)
    print()
    print(f"  Grid:       {grid_w} x {grid_h}")
    print(f"  Generator:  {grid_family}")
    print()
    print("Corrected protocol definitions:")
    print("  Non-nested: linear density schedule rho_i = (i+1)/N * rho_max,")
    print("              independent skeleton + obstacle seeds per env.")
    print("  Nested:     same density schedule, shared skeleton + shared")
    print("              obstacle-permutation seed.")
    print()

    # Test A
    test_builder_correctness(
        grid_w=grid_w, grid_h=grid_h,
        n_check_seeds=n_check_seeds, grid_family=grid_family,
    )

    # Test B
    df_b = test_f_star_sensibility(
        grid_w=grid_w, grid_h=grid_h,
        n_seeds=n_seeds, grid_family=grid_family,
    )
    df_b.to_csv(os.path.join(out_dir, 'test_b_f_star_sensibility.csv'),
                index=False)

    # Test C
    df_c = test_f_star_vs_densest_env(
        grid_w=grid_w, grid_h=grid_h,
        n_seeds=n_seeds, grid_family=grid_family,
    )
    df_c.to_csv(os.path.join(out_dir, 'test_c_f_star_vs_densest_env.csv'),
                index=False)

    # Summary
    print("=" * 64)
    print("PHASE 1 SUMMARY")
    print("=" * 64)
    print()
    print("Files saved to:")
    print(f"  {os.path.join(out_dir, 'test_b_f_star_sensibility.csv')}")
    print(f"  {os.path.join(out_dir, 'test_c_f_star_vs_densest_env.csv')}")
    print()
    print("What to look for in the output:")
    print("  Test A:")
    print("    - Per-env densities should track the schedule rho_i =")
    print("      (i+1)/N * rho_max for both modes.")
    print("    - Nested inclusion fraction should be 1.0 (or fall back")
    print("      to manual nested construction if generator doesn't")
    print("      have monotonic-prefix property).")
    print("    - Non-nested cross-env Jaccard should be SMALL (~rho^2)")
    print("      consistent with independent obstacle samples.")
    print("  Test B:")
    print("    - phi^nested and phi^non_nested at matched (rho_max, N)")
    print("      should be CLOSE (within ~5%).")
    print("    - Both should be in the range [0.55, 0.85] for cells in")
    print("      the FI transition window.")
    print("  Test C:")
    print("    - F*/F_nu_N ratio should be close to 1.0 in BOTH protocols.")
    print("    - Nested ratio = 1.0 exactly (by construction).")
    print("    - Non-nested ratio quantifies the extra shrinkage from")
    print("      independent lower-density envs.")
    print()
    print("If Test C shows non-nested ratio MUCH less than 1.0 (say < 0.7),")
    print("then F* is NOT primarily determined by the densest env in")
    print("non-nested mode, and the 'F* approx F_nu_N' claim is wrong.")
    print("This would mean nested and non-nested put F* at DIFFERENT")
    print("locations even at matched (rho_max, N), and the cross-protocol")
    print("bridge would need fundamentally different framing.")

    return {'test_b': df_b, 'test_c': df_c}


if __name__ == '__main__':
    main()
    