"""
phase3_non_nested_calibration_and_sweep.py

Phase 3: corrected non-nested protocol — calibration + targeted sweep,
matched to the shared-skel cells by realized phi.

Protocol definitions (recap)
----------------------------
All three protocols share the linear density schedule
   rho_i = (i+1)/N * rho_max,  i = 0 .. N-1.

What differs is the correlation structure across envs:

  Nested      : shared skeleton, shared obstacle-permutation
                -> O_nu_1 subset O_nu_2 subset ... subset O_nu_N
                   and F* = F_nu_N exactly.
  Shared-skel : shared skeleton, INDEPENDENT obstacle seeds per env.
                -> Same corridor structure, obstacles independently
                   sampled within it.  F* approx 0.80 * F_nu_N.
  Non-nested  : INDEPENDENT skeleton AND obstacle seeds per env.
                -> Different corridor structures per env.
                   F* substantially smaller than F_nu_N (ratio ~ 0.52-0.67).

Phase 2 measured shared-skel.  This script measures non-nested at the
SAME realized phi values shared-skel hit, so the three protocols can be
compared at the same nine (phi, N) targets.

Goal
----
Locate the third point on the correlation hierarchy:
   beta_F^non_nested, beta_N^non_nested

Predictions from the disagreement-variance mechanism:
   |beta_N^non_nested| > |beta_N^shared_skel| > |beta_N^nested|
   beta_F^non_nested  < beta_F^shared_skel  < beta_F^nested

Two-step pipeline
-----------------
STEP 3a -- CALIBRATION (pure geometry, ~5 minutes):
   Sweep rho_max across a fine grid at N in {2, 5, 10} in NON-NESTED
   mode, measure realized phi over 20 master_seeds per cell.

STEP 3b -- CELL SELECTION (instant):
   For each of the 9 target phi values that shared-skel hit, find the
   non-nested (rho_max, N) cell whose realized phi matches most closely.
   This produces 9 cells at the same realized phi as the shared-skel
   sweep, enabling three-way comparison at matched (phi, N).

STEP 3c -- TARGETED SWEEP (training-heavy):
   Train Q-tables for the chosen non-nested cells at n=50 seeds, score
   FI, fit the joint (phi, N) logistic surface.

Note on rho_max range
---------------------
Because non-nested F* is substantially SMALLER than nested or shared-skel
F* at the same (rho_max, N) (ratio 0.52-0.67), non-nested needs LOWER
rho_max values to reach the same realized phi as shared-skel.  The
calibration grid extends down to rho_max = 0.05 to ensure all 9 target
phi values are reachable.

Usage
-----
    python3 phase3_non_nested_calibration_and_sweep.py --step calibrate

    cat phase3_non_nested/chosen_cells.csv

    # Then re-run with --step all to launch training
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_dense_sweep_N_effect import analyze_dense_sweep
from fixed_rho_sweep_analyze import analyze_results
from targeted_f_star_sweep import aggregate_success_rate, analyze_targeted_sweep, plot_A_success_vs_N, plot_B_collapse, run_targeted_experiment


# ==============================================================
# Constants: protocol coefficients from the previous sweeps
# ==============================================================
GRID_W, GRID_H = 60, 30

# Nested re-run regression coefficients (60x30)
NESTED_ALPHA  = -22.93
NESTED_BETA_F = +33.20
NESTED_BETA_N = -0.21

# Shared-skel sweep regression coefficients (60x30)
SHARED_SKEL_ALPHA  = -18.75
SHARED_SKEL_BETA_F = +25.71
SHARED_SKEL_BETA_N = -0.288

# Target realized phi values from the shared-skel sweep
# (one row per (rho_max_chosen, N) cell that shared-skel actually ran)
SHARED_SKEL_TARGET_PHIS = [
    # (target_phi, N, comment)
    (0.633, 2,  'matched nested deep-failure (0.1)'),
    (0.693, 2,  'matched nested midpoint (0.5)'),
    (0.763, 2,  'matched nested onset (0.9)'),
    (0.653, 5,  'matched nested deep-failure (0.1)'),
    (0.708, 5,  'matched nested midpoint (0.5)'),
    (0.781, 5,  'matched nested onset (0.9)'),
    (0.684, 10, 'matched nested deep-failure (0.1)'),
    (0.737, 10, 'matched nested midpoint (0.5)'),
    (0.804, 10, 'matched nested onset (0.9)'),
]


# ==============================================================
# Family builder (must match Phase 1's build_family_corrected)
# ==============================================================

def build_family_corrected(
    rho, N, mode, master_seed, grid_w, grid_h,
    lambda_cliff=100.0, grid_family='maze',
    nested_schedule='linear',
):
    """Builds families under the CORRECTED protocol definitions.

    Same code as Phase 1 / Phase 2.  Only the 'non_nested' branch is
    exercised here.
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

    if nested_schedule == 'linear':
        rho_schedule = [round(((i + 1) / N) * rho, 4) for i in range(N)]
    elif nested_schedule == 'equal':
        rho_schedule = [rho] * N
    else:
        raise ValueError(f"unknown nested_schedule {nested_schedule!r}")

    envs = []
    if mode == 'shared_skel':
        shared_skel = master_seed & 0x7FFFFFFF
        for i, rho_i in enumerate(rho_schedule):
            obs_i = hash((master_seed, 'obs', i)) & 0x7FFFFFFF
            env = fam.make(
                size=size_spec, scale=1, walls_perc=rho_i,
                seed=shared_skel, obstacle_seed=obs_i, verbose=False,
            )
            env.name = (f'{env.name}-shared_skel-rhomax{rho:.3f}-N{N}-'
                        f'ms{master_seed}-i{i}-rho{rho_i:.3f}')
            envs.append(env)
    elif mode == 'nested':
        shared_skel = master_seed & 0x7FFFFFFF
        shared_obs  = hash((master_seed, 'nested_obs')) & 0x7FFFFFFF
        for i, rho_i in enumerate(rho_schedule):
            env = fam.make(
                size=size_spec, scale=1, walls_perc=rho_i,
                seed=shared_skel, obstacle_seed=shared_obs, verbose=False,
            )
            env.name = (f'{env.name}-nested-{nested_schedule}-'
                        f'rhomax{rho:.3f}-N{N}-ms{master_seed}-'
                        f'i{i}-rho{rho_i:.3f}')
            envs.append(env)
    elif mode in ('non_nested', 'iid'):
        # Independent skeleton AND obstacle seed per env
        for i, rho_i in enumerate(rho_schedule):
            seed_i = hash((master_seed, 'skel', i)) & 0x7FFFFFFF
            obs_i  = hash((master_seed, 'obs',  i)) & 0x7FFFFFFF
            env = fam.make(
                size=size_spec, scale=1, walls_perc=rho_i,
                seed=seed_i, obstacle_seed=obs_i, verbose=False,
            )
            env.name = (f'{env.name}-non_nested-ms{master_seed}-'
                        f'i{i}-rho{rho_i:.3f}')
            envs.append(env)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return envs, rho_schedule


def compute_f_star(envs, grid_w, grid_h):
    """Realized F*-density."""
    free_mask = np.ones((grid_w, grid_h), dtype=bool)
    for env in envs:
        env_walls = np.zeros((grid_w, grid_h), dtype=bool)
        for (x, y) in env.walls:
            if 0 <= x < grid_w and 0 <= y < grid_h:
                env_walls[x, y] = True
        free_mask &= ~env_walls
    return float(free_mask.sum() / (grid_w * grid_h))


# ==============================================================
# STEP 3a: non-nested calibration
# ==============================================================

def step_3a_calibration(
    rho_max_grid=None, N_values=(2, 5, 10),
    n_seeds=20, grid_w=GRID_W, grid_h=GRID_H,
    grid_family='maze',
):
    """
    Build NON-NESTED families across a fine rho_max grid at each N,
    measure realized phi.  Returns a DataFrame.
    """
    if rho_max_grid is None:
        # Non-nested needs LOWER rho_max to reach the same realized phi
        # as nested or shared-skel, because its F* is substantially
        # smaller (Phase 1 ratio 0.52-0.67).  Extend down to 0.05.
        rho_max_grid = np.round(np.arange(0.05, 0.96, 0.05), 3).tolist()

    rows = []
    print(f"Calibrating NON-NESTED phi across "
          f"{len(rho_max_grid)} rho_max values x {len(N_values)} N values "
          f"x {n_seeds} seeds...")
    total = len(rho_max_grid) * len(N_values)
    with tqdm(total=total, desc='calibration') as pbar:
        for N in N_values:
            for rho_max in rho_max_grid:
                phis = []
                for seed in range(n_seeds):
                    envs, _ = build_family_corrected(
                        rho=rho_max, N=N, mode='non_nested',
                        master_seed=seed, grid_w=grid_w, grid_h=grid_h,
                        grid_family=grid_family,
                    )
                    phis.append(compute_f_star(envs, grid_w, grid_h))
                rows.append({
                    'rho_max': rho_max, 'N': N,
                    'phi_mean': float(np.mean(phis)),
                    'phi_sd':   float(np.std(phis)),
                    'n_seeds':  len(phis),
                })
                pbar.update(1)
    return pd.DataFrame(rows)


# ==============================================================
# STEP 3b: cell selection matched to shared-skel target phis
# ==============================================================

def find_closest_rho_max(calibration_df, target_phi, N, max_tolerance=0.05):
    """
    Given the calibration DataFrame, find the rho_max value at family
    size N whose realized phi is closest to target_phi.  Returns None
    if no cell is within max_tolerance of target_phi.
    """
    candidates = calibration_df[calibration_df['N'] == N].copy()
    if len(candidates) == 0:
        return None
    candidates['phi_distance'] = (candidates['phi_mean'] - target_phi).abs()
    best = candidates.loc[candidates['phi_distance'].idxmin()]
    if best['phi_distance'] > max_tolerance:
        return None
    return {
        'rho_max':       float(best['rho_max']),
        'N':             int(N),
        'phi_realized':  float(best['phi_mean']),
        'phi_target':    float(target_phi),
        'phi_distance':  float(best['phi_distance']),
    }


def step_3b_pick_cells_matched_to_shared_skel(calibration_df,
                                               max_tolerance=0.05):
    """Pick non-nested cells matching shared-skel's 9 realized phi values.

    Each shared-skel cell has a (phi_realized, N).  We find the
    non-nested (rho_max, N) cell whose realized phi is closest to that
    target.  This produces a 9-cell shared-skel-matched design.
    """
    chosen = []
    for (target_phi, N, comment) in SHARED_SKEL_TARGET_PHIS:
        pick = find_closest_rho_max(
            calibration_df, target_phi, N, max_tolerance=max_tolerance,
        )
        if pick is not None:
            pick['comment'] = comment
            chosen.append(pick)
        else:
            print(f"  WARN: no non-nested cell within {max_tolerance:.2f} "
                  f"of phi={target_phi:.3f} at N={N} ({comment})")
            # Still include the closest miss so the user can decide
            # whether the gap is too large.
            candidates = calibration_df[calibration_df['N'] == N].copy()
            if len(candidates) > 0:
                candidates['phi_distance'] = \
                    (candidates['phi_mean'] - target_phi).abs()
                closest = candidates.loc[candidates['phi_distance'].idxmin()]
                print(f"    Closest available: "
                      f"rho_max={closest['rho_max']:.3f}, "
                      f"phi={closest['phi_mean']:.4f} "
                      f"(distance {closest['phi_distance']:.4f})")
    return pd.DataFrame(chosen)


# ==============================================================
# Step 3c: hand off to the targeted-sweep harness
# ==============================================================

def step_3c_build_chosen_cells_for_harness(chosen_cells_df):
    """Convert chosen cells to harness format."""
    return [
        (float(row['rho_max']), int(row['N']),
         round(float(row['phi_realized']), 3))
        for _, row in chosen_cells_df.iterrows()
    ]


# ==============================================================
# Main: calibration + cell selection (no training)
# ==============================================================

def main(out_dir='./phase3_non_nested/',
         grid_w=GRID_W, grid_h=GRID_H,
         calibration_seeds=20, grid_family='maze',
         max_tolerance=0.05):
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("#" * 64)
    print("# Phase 3 (steps 3a + 3b): non-nested calibration + cell pick")
    print("#" * 64)
    print()
    print(f"  Grid:                  {grid_w} x {grid_h}")
    print(f"  Generator:             {grid_family}")
    print(f"  Calibration seeds:     {calibration_seeds}")
    print(f"  Target phi values:     {len(SHARED_SKEL_TARGET_PHIS)} "
          f"(matched to shared-skel)")
    print(f"  phi-match tolerance:   {max_tolerance:.3f}")
    print()
    print("Recap: non-nested has independent skeleton AND obstacle seeds")
    print("per env.  F* is substantially smaller than F_nu_N (ratio 0.52-0.67),")
    print("so non-nested needs LOWER rho_max than shared-skel to reach the")
    print("same realized phi.")
    print()

    # Step 3a
    print("--- Step 3a: non-nested phi calibration ---")
    cal_df = step_3a_calibration(
        n_seeds=calibration_seeds,
        grid_w=grid_w, grid_h=grid_h,
        grid_family=grid_family,
    )
    cal_path = os.path.join(out_dir, 'calibration.csv')
    cal_df.to_csv(cal_path, index=False)
    print(f"Saved calibration table to {cal_path}")
    print()

    # Step 3b
    print("--- Step 3b: cell selection (matched to shared-skel) ---")
    chosen_df = step_3b_pick_cells_matched_to_shared_skel(
        cal_df, max_tolerance=max_tolerance,
    )
    chosen_path = os.path.join(out_dir, 'chosen_cells.csv')
    chosen_df.to_csv(chosen_path, index=False)
    print()
    print(f"Chosen {len(chosen_df)} non-nested cells:")
    print(chosen_df.to_string(index=False))
    print()

    # Step 3c estimate
    chosen_tuples = step_3c_build_chosen_cells_for_harness(chosen_df)
    total_envs = sum(int(row['N']) for _, row in chosen_df.iterrows())
    n_seeds_sweep = 50
    total_qtables = total_envs * n_seeds_sweep
    print()
    print("--- Step 3c: estimated compute for the targeted sweep ---")
    print(f"  Cells:           {len(chosen_df)}")
    print(f"  Envs per seed:   {total_envs}")
    print(f"  Seeds per cell:  {n_seeds_sweep}")
    print(f"  Total Q-tables:  {total_qtables}")
    print(f"  At ~30s per Q-table:  ~{total_qtables * 30 / 3600:.1f} hours")
    print()

    return {'calibration': cal_df, 'chosen_cells': chosen_df,
            'chosen_tuples': chosen_tuples}


# ==============================================================
# Entry point
# ==============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['calibrate', 'all'],
                        default='all')
    parser.add_argument('--calibration_seeds', type=int, default=20)
    parser.add_argument('--max_tolerance', type=float, default=0.05)
    args = parser.parse_args()


    run_dense_experiment = True # False # 
    just_train_qtables = False # True # 

    n_seeds = 50
    sizes = [(60, 30)]

    min_n_per_cell = n_seeds // 3

    test_name = 'phase3_non_nested'

    out_dir = f'./logs/{test_name}-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_csv     = f'{test_name}-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}.csv'

    calib_csv     = out_dir+f'{test_name}-calib-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}.csv'

    if os.path.exists(calib_csv):
        print(f"Loading existing calibration from {calib_csv}")
        calib_df = pd.read_csv(calib_csv, sep=';', decimal=',')
        print(calib_df)
        chosen_cells = step_3c_build_chosen_cells_for_harness(calib_df)
    else:
        calib_dic = main(calibration_seeds=args.calibration_seeds,
                         max_tolerance=args.max_tolerance)
        
        chosen_cells = calib_dic['chosen_tuples']
        calib_df = calib_dic['chosen_cells']
        calib_df.to_csv(calib_csv, sep=';', decimal=',')
        sys.exit(0)

    if args.step == 'calibrate':
        print()
        print("Calibration only.  To launch the targeted sweep, "
              "re-run with --step all.")
        sys.exit(0)

    if not os.path.exists(out_dir+output_csv):

        result = run_targeted_experiment(
            size               = sizes[0], # (60, 30),
            chosen_cells       = chosen_cells,
            n_seeds            = n_seeds, # 5, # 30,
            mode               = 'non_nested',   # <-- the key change
            lambda_cliff       = 100.0,
            grid_family        = 'maze',
            out_dir            = out_dir,
            output_csv         = output_csv,
            just_train_qtables=just_train_qtables,
        )

        sweep_df = result['sweep_df']
                                
    else:
        sweep_df = pd.read_csv(out_dir+output_csv).reset_index()

    if not just_train_qtables:

        summary = analyze_targeted_sweep(
            sweep_df,
            out_dir='./logs/targeted_sweep/',
            min_n_per_cell=min_n_per_cell,
        )

        grouped = aggregate_success_rate(sweep_df)

        for size in sizes:
            plot_A_success_vs_N(grouped, size_filter=f'{size[0]}x{size[1]}', fig_name=out_dir+f'fixed_rho_sweep-A_success_vs_N-{size[0]}x{size[1]}.png')

            plot_B_collapse(sweep_df, size_filter=f'{size[0]}x{size[1]}', fig_name=out_dir+f'fixed_rho_sweep-B_collapse-{size[0]}x{size[1]}.png')

        analyze_results(out_dir+output_csv)

        # sweep_df = pd.read_csv('./logs/targeted_dense_sweep-60x30-60x30-S=1/targeted_dense_sweep-60x30-60x30-S=1.csv')
        summary = analyze_dense_sweep(
            sweep_df,
            out_dir=out_dir+'/dense_analysis/',
        )
        print(f"N-effect p-value: {summary['fit']['N_effect']['p']:.4f}")
        print(f"Transition width: {4.4/abs(summary['fit']['F_slope']['estimate']):.3f}")        

    print()
        