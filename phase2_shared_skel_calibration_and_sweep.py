"""
phase2_shared_skel_calibration_and_sweep.py

Phase 2: shared-skel protocol, designed for cross-protocol bridge testing
against the existing nested re-run.

Background
----------
Phase 1 established three findings:
  1. The three protocols (nested, shared-skel, non-nested) form a clean
     correlation hierarchy.
  2. Non-nested F* is substantially below F_nu_N (ratio 0.52-0.67), so
     non-nested and nested CANNOT be compared at matched (rho_max, N) --
     the protocols put F* at different locations.
  3. Shared-skel F*-density saturates toward 1 - f_skel ~ 0.543 as N
     grows or rho_max approaches 1. Its operational window is roughly
     phi in [0.54, 0.72].

The corrected cross-protocol bridge framework compares protocols at
MATCHED REALIZED phi, not at matched (rho_max, N).  This file produces
the data needed for that comparison on shared-skel.

Two-step pipeline
-----------------
STEP 2a -- CALIBRATION (pure geometry, ~5 minutes):
   Sweep rho_max across a fine grid at N in {2, 5, 10}, measure realized
   phi over 20 master_seeds per cell.  Produces a lookup table that
   maps desired phi to the rho_max needed in shared-skel mode.

STEP 2b -- CELL SELECTION (pure geometry, instant):
   Using the calibration table, pick the shared-skel (rho_max, N) cells
   that match the nested re-run's phi values most closely.  Defaults to
   matching cells at the nested critical region (phi ~ 0.62-0.74).

STEP 2c -- TARGETED SWEEP (training-heavy, hours to ~1 day):
   Train Q-tables for the chosen cells at n=50 seeds, score FI, and fit
   the joint (phi, N) logistic surface.  Output ready for direct
   comparison with the nested re-run.

Output of Phase 2 will be the regression coefficients
    beta_F^{shared_skel}, beta_N^{shared_skel}
plus per-cell P(FI) values for direct nested-vs-shared-skel comparison
at matched realized phi.

Usage
-----
    # Step 2a + 2b: calibrate and pick cells
    python3 phase2_shared_skel_calibration_and_sweep.py --step calibrate

    # Inspect the chosen cells before launching training
    cat phase2_shared_skel/chosen_cells.csv

    # Step 2c: hand the chosen cells to run_targeted_experiment
    # (see the example at the bottom of this file)
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from analyze_dense_sweep_N_effect import analyze_dense_sweep
from fixed_rho_sweep_analyze import analyze_results
from targeted_f_star_sweep import aggregate_success_rate, analyze_targeted_sweep, plot_A_success_vs_N, plot_B_collapse, run_targeted_experiment


# ==============================================================
# Constants from Phase 1 / the nested re-run
# ==============================================================
GRID_W, GRID_H = 60, 30

# Nested re-run regression coefficients (60x30)
NESTED_ALPHA  = -22.93
NESTED_BETA_F = +33.20
NESTED_BETA_N = -0.21

# Nested re-run cells and their realized phi values (from Phase 1 Test B)
NESTED_PHI_AT_CELL = {
    (0.60, 2):  0.7259,
    (0.60, 5):  0.7259,
    (0.60, 10): 0.7259,
    (0.70, 2):  0.6802,
    (0.70, 5):  0.6802,
    (0.70, 10): 0.6802,
    (0.80, 5):  0.6344,
}

# Nested critical (median) threshold per N:
#   phi_c^(0.5)(N) = (-alpha - beta_N * N) / beta_F
def nested_phi_critical(N, level=0.5):
    """Return the phi at which P(FI) = level on the nested surface."""
    from math import log
    logit = log(level / (1 - level))
    return (logit - NESTED_ALPHA - NESTED_BETA_N * N) / NESTED_BETA_F


# ==============================================================
# Family builder (must match Phase 1's build_family_corrected)
# ==============================================================

def build_family_corrected(
    rho, N, mode, master_seed, grid_w, grid_h,
    lambda_cliff=100.0, grid_family='maze',
    nested_schedule='linear',
):
    """Same as Phase 1 (see phase1_corrected_protocol_geometry_check.py)."""
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
# STEP 2a: calibration
# ==============================================================

def step_2a_calibration(
    rho_max_grid=None, N_values=(2, 5, 10),
    n_seeds=20, grid_w=GRID_W, grid_h=GRID_H,
    grid_family='maze',
):
    """
    Build shared-skel families across a fine rho_max grid at each N,
    measure realized phi.  Returns a DataFrame with one row per
    (rho_max, N) cell.
    """
    if rho_max_grid is None:
        # Fine grid in the operational window
        rho_max_grid = np.round(np.arange(0.10, 0.96, 0.05), 3).tolist()

    rows = []
    print(f"Calibrating shared-skel phi across "
          f"{len(rho_max_grid)} rho_max values x {len(N_values)} N values "
          f"x {n_seeds} seeds...")
    total = len(rho_max_grid) * len(N_values)
    with tqdm(total=total, desc='calibration') as pbar:
        for N in N_values:
            for rho_max in rho_max_grid:
                phis = []
                for seed in range(n_seeds):
                    envs, _ = build_family_corrected(
                        rho=rho_max, N=N, mode='shared_skel',
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
# STEP 2b: cell selection
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


def step_2b_pick_cells(
    calibration_df, n_per_N=3, phi_grid=None, N_values=(2, 5, 10),
):
    """
    Pick shared-skel cells that cover the critical region and match
    nested re-run phi values where possible.

    Default cell strategy (n_per_N=3):
      For each N, pick one cell at the nested median threshold (the
      "matched-to-critical" point), one near the lower edge of the
      operational window, and one near the upper edge.  Three cells
      per N x 3 N values = 9 cells total.

    If phi_grid is provided, use that as the target phi grid instead.
    """
    if phi_grid is None:
        chosen = []
        for N in N_values:
            phi_crit = nested_phi_critical(N, level=0.5)
            # Lower edge: nested phi at deep-failure (P=0.1)
            phi_low  = nested_phi_critical(N, level=0.1)
            # Upper edge: nested phi at onset (P=0.9)
            phi_high = nested_phi_critical(N, level=0.9)
            targets = [phi_low, phi_crit, phi_high][:n_per_N]
            for tgt in targets:
                pick = find_closest_rho_max(calibration_df, tgt, N)
                if pick is not None:
                    chosen.append(pick)
        return pd.DataFrame(chosen)
    else:
        chosen = []
        for N in N_values:
            for tgt in phi_grid:
                pick = find_closest_rho_max(calibration_df, tgt, N)
                if pick is not None:
                    chosen.append(pick)
        return pd.DataFrame(chosen)


# ==============================================================
# Step 2c: targeted sweep config (training delegated to user's harness)
# ==============================================================

def step_2c_build_chosen_cells_for_harness(chosen_cells_df):
    """
    Convert the chosen-cells DataFrame into the (rho, N, F_target)
    tuple list that run_targeted_experiment expects via chosen_cells.
    """
    return [
        (float(row['rho_max']), int(row['N']), round(float(row['phi_realized']), 3))
        for _, row in chosen_cells_df.iterrows()
    ]


# ==============================================================
# Main driver
# ==============================================================

def main(out_dir='./phase2_shared_skel/',
         grid_w=GRID_W, grid_h=GRID_H,
         calibration_seeds=20, grid_family='maze',
         n_per_N=3):
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("#" * 64)
    print("# Phase 2 (steps 2a + 2b): shared-skel calibration + cell pick")
    print("#" * 64)
    print()
    print(f"  Grid:              {grid_w} x {grid_h}")
    print(f"  Generator:         {grid_family}")
    print(f"  Calibration seeds: {calibration_seeds}")
    print(f"  Cells per N:       {n_per_N}")
    print()

    # Step 2a
    print("--- Step 2a: shared-skel phi calibration ---")
    cal_df = step_2a_calibration(
        n_seeds=calibration_seeds,
        grid_w=grid_w, grid_h=grid_h,
        grid_family=grid_family,
    )
    cal_path = os.path.join(out_dir, 'calibration.csv')
    cal_df.to_csv(cal_path, index=False)
    print(f"Saved calibration table to {cal_path}")
    print()

    # Step 2b
    print("--- Step 2b: cell selection ---")
    print(f"\nNested critical thresholds (level=0.5):")
    for N in [2, 5, 10]:
        print(f"  N={N}: phi_c^(0.5) = {nested_phi_critical(N, 0.5):.4f}, "
              f"phi_c^(0.9) = {nested_phi_critical(N, 0.9):.4f}, "
              f"phi_c^(0.1) = {nested_phi_critical(N, 0.1):.4f}")
    print()

    chosen_df = step_2b_pick_cells(cal_df, n_per_N=n_per_N)
    chosen_path = os.path.join(out_dir, 'chosen_cells.csv')
    chosen_df.to_csv(chosen_path, index=False)
    print(f"Chosen {len(chosen_df)} shared-skel cells:")
    print(chosen_df.to_string(index=False))
    print()

    # Step 2c: compute estimate and harness-ready format
    chosen_tuples = step_2c_build_chosen_cells_for_harness(chosen_df)
    total_envs = sum(int(row['N']) for _, row in chosen_df.iterrows())
    n_seeds_sweep = 50  # match nested re-run
    total_qtables = total_envs * n_seeds_sweep
    print()
    print("--- Step 2c: estimated compute for the targeted sweep ---")
    print(f"  Cells:           {len(chosen_df)}")
    print(f"  Envs per seed:   {total_envs}")
    print(f"  Seeds per cell:  {n_seeds_sweep}")
    print(f"  Total Q-tables:  {total_qtables}")
    print(f"  At ~30s per Q-table (your retrained rate):")
    print(f"    -> ~{total_qtables * 30 / 3600:.1f} hours wall-clock")
    print()
    print("To launch step 2c, hand chosen_cells.csv values to "
          "run_targeted_experiment via chosen_cells=...")
    print("Example:")
    print("    chosen_cells = step_2c_build_chosen_cells_for_harness(chosen_df)")
    print("    run_targeted_experiment(")
    print("        size=(60, 30),")
    print("        mode='shared_skel',")
    print("        chosen_cells=chosen_cells,")
    print("        n_seeds=50,")
    print("        ...")
    print("    )")
    print()
    return {'calibration': cal_df, 'chosen_cells': chosen_df, 'chosen_tuples': chosen_tuples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['calibrate', 'all'],
                        default='all',
                        help='Which step to run.  "calibrate" runs '
                             'only 2a+2b (no training).')
    parser.add_argument('--n_per_N', type=int, default=3,
                        help='Number of cells per N value (default 3)')
    parser.add_argument('--calibration_seeds', type=int, default=20)
    args = parser.parse_args()


    run_dense_experiment = True # False # 
    just_train_qtables = False # True # 

    n_seeds = 50
    sizes = [(60, 30)]

    min_n_per_cell = n_seeds // 3

    test_name ='phase2_shared_skel'

    out_dir = f'./logs/{test_name}-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_csv     = f'{test_name}-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}.csv'

    calib_csv     = out_dir+f'{test_name}-calib-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}.csv'

    if os.path.exists(calib_csv):
        print(f"Loading existing calibration from {calib_csv}")
        calib_df = pd.read_csv(calib_csv, sep=';', decimal=',')
        chosen_cells = step_2c_build_chosen_cells_for_harness(calib_df)
    else:
        calib_dic = main(n_per_N=args.n_per_N,
            calibration_seeds=args.calibration_seeds)
        
        chosen_cells = calib_dic['chosen_tuples']
        calib_df = calib_dic['chosen_cells']
        calib_df.to_csv(calib_csv, sep=';', decimal=',')

    if not os.path.exists(out_dir+output_csv):

        result = run_targeted_experiment(
            size               = sizes[0], # (60, 30),
            chosen_cells       = chosen_cells,
            n_seeds            = n_seeds, # 5, # 30,
            mode               = 'shared_skel',
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
        