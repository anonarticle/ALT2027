"""
fixed_rho_sweep.py

Experiment for distinguishing voting vs. percolation as the operating
mechanism in non-nested Free Inference.

Sweeps (rho, N, seed, mode) and records:
  - geometric:  |F*|, F*-density, min s0->g cut in F*
  - coherence:  C(V), interpretation flag
  - agent-side: VM return / collisions / bps, PMS collisions
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import math

from GridWorldCanonicalBenchmarkSuite import RandomMazeFamily, RandomScatterFamily, SizeSpec
from MultiAgentMainRecupE import load_qtable
from fixed_rho_sweep_analyze import analyze_results
from fixed_rho_sweep_helpers import run_vm_pms_rollouts, train_qtables_for_envs
from navigational_coherence_new import (
    _identify_consensus_free_space,
    compute_q_disagreement,
    compute_coherence_score,
)
from analyze_dense_sweep_N_effect import analyze_dense_sweep

from hashlib import sha256
def hash(s):
    h = sha256()
    h.update(str(s).encode('utf-8'))
    return int(h.hexdigest(), 16)

# ---------------------------------------------------------------
# 1. Suite builder for the fixed-rho replicated draw.
# ---------------------------------------------------------------
# Your existing build_canonical_suite is parameterised by
# (perc0, percN, num_percs, nested_grids).  We need a variant that
# instead takes (rho_fixed, N, mode, master_seed) and returns
# N environments all at the same density rho_fixed, with obstacle
# patterns drawn either i.i.d. (mode='iid') or sharing a maze
# skeleton (mode='shared_skel').
#
# The cleanest way is to add a thin wrapper that calls your
# canonical builder N times with carefully chosen seeds.

def build_fixed_rho_family0(size, rho, N, mode, master_seed, 
                           grid_family='maze',          # 'maze' | 'scatter'
                           lambda_cliff=100.0, **kwargs):
    """
    Returns a list of N gridworld envs, all at density rho, on the
    given grid size.

      mode = 'iid'         : each env gets an independent seed
      mode = 'shared_skel' : a single 'skeleton' seed is reused for
                             maze-corridor generation, only the
                             obstacle pattern varies per env

    The exact mechanism for splitting 'skeleton' from 'obstacle
    pattern' depends on how build_canonical_suite is implemented;
    if it doesn't already separate them, the simplest fallback is:
      - 'iid'         : seed_i = hash((master_seed, i))
      - 'shared_skel' : seed_i = master_seed for all i
    and rely on the maze post-processing step (start-goal connectivity
    enforcement) to introduce the skeleton correlation naturally.
    """
    if grid_family == 'maze':
        fam = RandomMazeFamily(C=10.0, lambda_cliff=lambda_cliff)
    elif grid_family == 'scatter':
        fam = RandomScatterFamily(C=10.0, lambda_cliff=lambda_cliff,
                                  connectivity_mode='carve')
    else:
        raise ValueError(f"unknown grid_family {grid_family!r}")
    size_spec = SizeSpec(size[0], size[1])

    envs = []
    for i in range(N):
        if mode == 'iid':
            # seed_i = int(np.random.SeedSequence([master_seed, i]).generate_state(1)[0])
            seed_i = hash((master_seed, i)) & 0x7FFFFFFF
        elif mode == 'shared_skel':
            seed_i = master_seed & 0x7FFFFFFF
        else:
            raise ValueError(f"unknown mode {mode!r}")

        env = fam.make(
            size=size_spec, scale=1, walls_perc=rho,
            seed=seed_i, obstacle_seed=hash((master_seed,i)) & 0x7FFFFFFF, verbose=False,
        )

        # Make the cache key unique per (mode, master_seed, i).
        env.name = f'{env.name}-{mode}-ms{master_seed}-i{i}'
        envs.append(env)        

    return envs

def build_fixed_rho_family1(size, rho, N, mode, master_seed,
                           grid_family='maze',          # 'maze' | 'scatter'
                           lambda_cliff=100.0,
                           nested_schedule='linear',    # used only when mode='nested'
                           **kwargs):
    """
    Returns a list of N gridworld envs on the given grid size.

    Modes
    -----
    mode = 'iid'         : each env gets an independent seed; per-env
                           obstacle density = rho.  F*-density typically
                           shrinks as ~(1-rho)^N at fixed rho (modulated
                           by generator correlations).

    mode = 'shared_skel' : a single 'skeleton' seed is reused for
                           maze-corridor generation, only the obstacle
                           pattern varies per env.  Per-env density = rho.

    mode = 'nested'      : all envs share the same maze skeleton AND a
                           common obstacle-permutation, but each env's
                           obstacle set is a PREFIX of that permutation
                           at a different density:
                                rho_i = schedule[i] * rho_max,  i=1..N
                           where rho here is interpreted as rho_max
                           (the densest env's density) and the schedule
                           is determined by `nested_schedule`.
                           This guarantees O_{nu_1} subset O_{nu_2} subset ... O_{nu_N},
                           the defining property of a nested family.
                           F* = F_{nu_N} (the densest env), so F*-density
                           depends only on rho_max, not on N.

    Parameters
    ----------
    rho : float
        For 'iid' and 'shared_skel': per-environment obstacle density.
        For 'nested': the MAXIMUM density in the family (rho_max);
                      env i has density rho_i = schedule[i] * rho.

    nested_schedule : str, default 'linear'
        Only used when mode == 'nested'.
        'linear': rho_i = ((i + 1) / N) * rho_max for i = 0..N-1.
                  This evenly spaces densities from rho_max/N up to
                  rho_max.  This is the standard nested protocol.
        'equal' : rho_i = rho_max for all i.  All envs share the same
                  density; useful as a degenerate sanity check that
                  the nested mechanism reduces to a single-env case
                  when the schedule is trivial.
    """
    if grid_family == 'maze':
        fam = RandomMazeFamily(C=10.0, lambda_cliff=lambda_cliff)
    elif grid_family == 'scatter':
        fam = RandomScatterFamily(C=10.0, lambda_cliff=lambda_cliff,
                                  connectivity_mode='carve')
    else:
        raise ValueError(f"unknown grid_family {grid_family!r}")

    size_spec = SizeSpec(size[0], size[1])

    if mode in ('iid', 'shared_skel'):
        # Existing behaviour (unchanged).
        envs = []
        for i in range(N):
            if mode == 'iid':
                seed_i = hash((master_seed, i)) & 0x7FFFFFFF
            else:  # shared_skel
                seed_i = master_seed & 0x7FFFFFFF
            env = fam.make(
                size=size_spec, scale=1, walls_perc=rho,
                seed=seed_i,
                obstacle_seed=hash((master_seed, i)) & 0x7FFFFFFF,
                verbose=False,
            )
            env.name = f'{env.name}-{mode}-ms{master_seed}-i{i}'
            envs.append(env)
        return envs, [rho] * N

    elif mode == 'nested':
        # Density schedule
        if nested_schedule == 'linear':
            rho_schedule = [round(((i + 1) / N) * rho, 3) for i in range(N)]
        elif nested_schedule == 'equal':
            rho_schedule = [rho] * N
        else:
            raise ValueError(
                f"unknown nested_schedule {nested_schedule!r}; "
                f"use 'linear' or 'equal'"
            )

        # Build each env with the SAME skeleton and obstacle seeds,
        # varying only walls_perc.  If fam.make's obstacle sampling at
        # rising walls_perc with fixed seeds is monotonic (the prefix
        # property), this is the cleanest nested construction.
        shared_maze_seed     = master_seed & 0x7FFFFFFF
        shared_obstacle_seed = hash((master_seed, 'nested_obs')) & 0x7FFFFFFF

        envs = []
        for i, rho_i in enumerate(rho_schedule):
            env = fam.make(
                size=size_spec, scale=1, walls_perc=rho_i,
                seed=shared_maze_seed,
                obstacle_seed=shared_obstacle_seed,
                verbose=False,
            )
            env.name = (
                f'{env.name}-nested-{nested_schedule}-'
                f'rhomax{rho:.3f}-N{N}-ms{master_seed}-i{i}-rho{rho_i:.3f}'
            )
            envs.append(env)

        # Verify nesting; if violated, fall back to manual prefix
        # construction.  This catches the case where fam.make does
        # NOT produce monotonic obstacle prefixes from rising
        # walls_perc.
        for i in range(1, N):
            o_prev = set(envs[i - 1].walls)
            o_curr = set(envs[i].walls)
            if not o_prev.issubset(o_curr):
                envs = _build_nested_family_manual(
                    fam, size_spec, rho_schedule,
                    master_seed, nested_schedule, rho,
                )
                break

        return envs, rho_schedule

    else:
        raise ValueError(f"unknown mode {mode!r}")

def build_fixed_rho_family(
    size, rho, N, mode, master_seed, 
    grid_family='maze',
    lambda_cliff=100.0, 
    nested_schedule='linear',
    full_env_name=True,
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
    grid_w, grid_h = size

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
            if full_env_name:
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
            if full_env_name:
                env.name = (f'{env.name}-nested-{nested_schedule}-'
                        f'rhomax{rho:.3f}-N{N}-ms{master_seed}-'
                        f'i{i}-rho{rho_i:.3f}')
            envs.append(env)
    elif mode in ('non_nested', 'iid'):
        for i, rho_i in enumerate(rho_schedule):
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
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return envs, rho_schedule

def _build_nested_family_manual(fam, size_spec, rho_schedule,
                                master_seed, nested_schedule, rho_max):
    """
    Manual nested-family construction, used as a fallback when
    fam.make does not produce monotonic obstacle prefixes.

    Strategy:
      1. Build a single "full" env at walls_perc = rho_max.
      2. Take its obstacle set as the candidate walls.
      3. Deterministically permute those walls (seeded by master_seed).
      4. For each rho_i in the schedule, take the first
         k_i = round(rho_i * n_full / rho_max) walls of that ordering
         as env i's obstacle set.  By construction the prefixes are
         strictly nested.
      5. Wrap each prefix into a fresh env: build at walls_perc=0 with
         the same skeleton seed, then override its `walls` attribute.
    """
    import numpy as np

    full_env = fam.make(
        size=size_spec, scale=1, walls_perc=rho_max,
        seed=master_seed & 0x7FFFFFFF,
        obstacle_seed=hash((master_seed, 'nested_obs_manual')) & 0x7FFFFFFF,
        verbose=False,
    )
    full_walls = list(full_env.walls)
    n_full = len(full_walls)

    rng = np.random.default_rng(master_seed & 0xFFFFFFFF)
    perm = rng.permutation(n_full)
    ordered_walls = [full_walls[j] for j in perm]

    envs = []
    for i, rho_i in enumerate(rho_schedule):
        k_i = int(round((rho_i / rho_max) * n_full)) if rho_max > 0 else 0
        k_i = max(0, min(k_i, n_full))
        walls_i = set(ordered_walls[:k_i])

        env = fam.make(
            size=size_spec, scale=1, walls_perc=0.0,
            seed=master_seed & 0x7FFFFFFF,
            obstacle_seed=hash((master_seed, 'nested_skel_only')) & 0x7FFFFFFF,
            verbose=False,
        )
        env.walls = walls_i
        if hasattr(env, '_recompute_islands'):
            env._recompute_islands()

        env.name = (
            f'{full_env.name.split("-")[0]}-nested-{nested_schedule}-manual-'
            f'rhomax{rho_max:.3f}-N{len(rho_schedule)}-'
            f'ms{master_seed}-i{i}-rho{rho_i:.3f}'
        )
        envs.append(env)

    return envs

# ---------------------------------------------------------------
# 2. The per-cell measurement.
# ---------------------------------------------------------------
def measure_one_realisation(envs, qtable_dir, agent_name='qlearning',
                            run_agent_rollouts=False, default_state_len=2):
    """
    Given a family `envs` (list of gridworld envs), returns a dict of
    measurements: |F*|, C(V), VM/PMS rollout stats (optional).
    """
    W, H = envs[0].width, envs[0].height
    start = (0, H - 1)
    goal  = (W - 1, 0)

    # -- load Q-tables ------------------------------------------------
    qs = []
    for env in envs:
        state_len = len(env.reset())
        q, _ = load_qtable(agent_name, env.name,
                           data_dir=qtable_dir,
                           state_len=state_len,
                           default_state_len=default_state_len,
                           suffix='')
        qs.append(q)

    # -- geometric: |F*| and connectivity -----------------------------
    F_star = _identify_consensus_free_space(qs, W, H, start, goal)
    F_dens = len(F_star) / (W * H)

    # path through F*?  reuse the coherence routine, which returns
    # C(V) = -inf when no path exists
    cv, path_info = compute_coherence_score(qs, W, H, start, goal)

    # -- agent-side rollouts (optional, slower) -----------------------
    rollout = {}
    if run_agent_rollouts:
        rollout = run_vm_pms_rollouts(envs, qs, start, goal)
        # ^ implement on top of your existing Meta-Agent runner;
        # returns dict with 'vm_return', 'vm_collisions', 'vm_bps_mean',
        # 'pms_collisions', etc.

    return {
        'F_star_size': len(F_star),
        'F_star_density': F_dens,
        'C_V': cv,
        'C_interp': classify_cv(cv),       # 'free', 'marginal', 'incoherent', 'no_path'
        **rollout,
    }


def classify_cv(cv):
    if cv == float('-inf'):
        return 'no_path'
    if cv < 0:
        return 'incoherent'
    if cv == 0:
        return 'marginal'
    return 'free'


"""
Targeted F*-density sweep: extension to fixed_rho_sweep.py.

Adds three functions to the existing infrastructure:
  - calibrate_N_for_F_star_targets : cheap geometric pre-sweep, no Q-tables
  - pick_targeted_cells            : given calibration, choose (rho, N) cells
  - run_targeted_sweep             : main loop, train Q-tables only on chosen cells

Plus:
  - analyze_targeted_sweep         : produces the decisive collapse plot and the
                                     F*-density-conditional cross-rho test

Drop-in placement: add this code below `run_sweep` in fixed_rho_sweep.py.
"""

# ===================================================================
# 1. Calibration pre-sweep: (rho, N) -> mean F*-density, no Q-tables
# ===================================================================

def calibrate_N_for_F_star_targets(
    size,
    rho_grid,
    candidate_Ns       = (1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 21),
    n_seeds            = 5,
    mode               = 'iid',
    lambda_cliff       = 100.0,
    grid_family        = 'maze',
    master_seed_base   = 0,
    show_progress      = True,
):
    """
    For each (rho, N) in (rho_grid x candidate_Ns), build n_seeds families
    and compute the mean F*-density GEOMETRICALLY (no Q-table training).

    F*-density is set-intersection of free cells across the N envs in a
    family — depends only on env.walls.  Variance across seeds at fixed
    (rho, N) is small (typical std ~0.005-0.01), so n_seeds=5 is enough
    to estimate the mean within ~0.005.

    Returns
    -------
    calibration : pd.DataFrame
        Columns: rho, N, F_dens_mean, F_dens_std, n_seeds.
        One row per (rho, N) candidate.
    """
    rows = []
    W, H = size
    start = (0, H - 1)
    goal  = (W - 1, 0)

    candidates = list(product(rho_grid, candidate_Ns))
    iterator = tqdm(candidates, desc='calibration') if show_progress else candidates

    for rho, N in iterator:
        f_stars = []
        for seed in range(n_seeds):
            # Use a master_seed that includes the size to keep nominal
            # independence across grid sizes that might share this code.
            master_seed = hash((master_seed_base, size, mode, rho, N, seed)) & 0xFFFFFFFF

            envs = build_fixed_rho_family(
                size=size, rho=rho, N=N,
                mode=mode, master_seed=master_seed,
                lambda_cliff=lambda_cliff,
                grid_family=grid_family,
            )

            # Direct geometric F*-density: no Q-tables needed.
            free_cells = {(x, y) for y in range(H) for x in range(W)
                          if (x, y) != start and (x, y) != goal}
            for env in envs:
                free_cells -= set(env.walls)
            # add start and goal back -- they're always considered free
            f_dens = (len(free_cells) + 2) / (W * H)
            f_stars.append(f_dens)

        rows.append({
            'rho': rho,
            'N':   N,
            'F_dens_mean': float(np.mean(f_stars)),
            'F_dens_std':  float(np.std(f_stars)),
            'n_seeds': n_seeds,
        })

    return pd.DataFrame(rows).sort_values(['rho', 'N']).reset_index(drop=True)


# ===================================================================
# 2. Target picker: choose (rho, N) per F*-density target
# ===================================================================

def pick_targeted_cells(
    calibration,
    F_star_targets = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40),
    max_abs_miss   = 0.05,   # warn if best (rho, N) misses target by more than this
    max_cells_per_F_star = 4,
    verbose        = True,
):
    """
    For each (rho, F*_target), pick the N whose calibrated mean F*-density
    is closest to the target.  Skip targets that no N can reach (e.g.,
    targets above the N=1 F*-density, or below the largest-N reach).

    Returns
    -------
    targeted : pd.DataFrame
        Columns: rho, N, F_target, F_dens_calibrated, miss.
        One row per (rho, F*_target) where a usable N was found.
    """
    rows = []
    for rho in sorted(calibration['rho'].unique()):
        rho_cal = calibration[calibration['rho'] == rho]
        for F_target in F_star_targets:

            # Closest N by absolute distance to target
            diffs = (rho_cal['F_dens_mean'] - F_target).abs()
            best_idx = diffs.idxmin()
            best = rho_cal.loc[best_idx]
            miss = float(best['F_dens_mean'] - F_target)

            if abs(miss) > max_abs_miss and verbose:
                print(f"  [target picker] rho={rho:.2f} target={F_target:.2f}: "
                      f"closest is N={int(best['N'])} at F*={best['F_dens_mean']:.3f} "
                      f"(miss = {miss:+.3f}, exceeds {max_abs_miss}); "
                      f"target may be outside the (rho, N) grid's reach.")

            rows.append({
                'rho': rho,
                'N':   int(best['N']),
                'F_target': F_target,
                'F_dens_calibrated': float(best['F_dens_mean']),
                'miss': miss,
            })

    targeted = pd.DataFrame(rows)

    # Deduplicate: if the same (rho, N) is chosen for multiple F*_targets
    # (which happens at the extremes), keep the row whose target is closest.
    targeted = (targeted
                .assign(abs_miss=targeted['miss'].abs())
                .sort_values('abs_miss')
                .drop_duplicates(subset=['rho', 'N'], keep='first')
                .drop(columns='abs_miss')
                .sort_values(['rho', 'F_target'])
                .reset_index(drop=True))
    
    filtered_targeted = pd.DataFrame()
    for F_target in sorted(targeted['F_target'].unique()):
        F_cal = targeted[targeted['F_target'] == F_target].sort_values(['miss'], key=abs)
        if isinstance(F_cal, pd.Series):
            F_cal = pd.DataFrame(F_cal).T
        filtered_targeted = pd.concat([filtered_targeted, F_cal.iloc[:max_cells_per_F_star]], ignore_index=True)

    targeted = filtered_targeted

    if verbose:
        print(f"Selected {len(targeted)} unique (rho, N) cells "
              f"from {len(F_star_targets) * calibration['rho'].nunique()} "
              f"target requests.")

    return targeted


# ===================================================================
# 3. Targeted main loop
# ===================================================================

def run_targeted_sweep(
    targeted_cells,
    size,
    n_seeds        = 30,
    mode           = 'iid',
    run_rollouts   = False,
    output_csv     = './logs/targeted_sweep/targeted.csv',
    qtable_root    = './logs/targeted_sweep/qtable',
    agent_name     = 'qlearning',
    lambda_cliff   = 100.0,
    grid_family    = 'maze',
    just_train_qtables = False,
):
    """
    Run the heavy Q-table-training loop on the (rho, N) cells chosen
    by pick_targeted_cells.  This is the main experiment.

    Parameters
    ----------
    targeted_cells : pd.DataFrame
        Output of pick_targeted_cells.  Required columns: rho, N,
        F_target, F_dens_calibrated.

    Outputs are the same CSV format as run_sweep, with two extra columns
    so the analysis can group both ways:
        F_target           : the target this cell was meant to hit
        F_dens_calibrated  : the calibration-predicted F*-density
    """
    rows = []
    W, H = size

    # The unit of work is (cell, seed).  Iterate in this order so the
    # tqdm bar reflects total compute.
    work_units = [
        (cell_row, seed)
        for _, cell_row in targeted_cells.iterrows()
        for seed in range(n_seeds)
    ]

    for cell_row, seed in tqdm(work_units, desc='targeted sweep'):
        rho        = float(cell_row['rho'])
        N          = int(cell_row['N'])
        F_target   = float(cell_row['F_target'])
        F_calib    = float(cell_row['F_dens_calibrated'])
        
        # n_seeds replications are independent.
        master_seed = hash((size, mode, rho, N, seed)) & 0xFFFFFFFF

        # 1. Build family (identical seeding scheme to run_sweep, so the
        # Q-table cache directory layout stays compatible if you happen
        # to land on a (rho, N, seed) that overlaps with an earlier
        # run_sweep call).
        envs = build_fixed_rho_family(
            size=size, rho=rho, N=N,
            mode=mode, master_seed=master_seed,
            lambda_cliff=lambda_cliff, 
            grid_family=grid_family, full_env_name=False
        )[0]

        # 2. Train (or load cached) Q-tables for the family
        qtable_dir = (f"{qtable_root}_rmaze{lambda_cliff:.1f}"
                      f"-rho{rho:.2f}-N{N}-{mode}-seed{master_seed}")
        if not os.path.exists(qtable_dir):
            os.makedirs(qtable_dir)

        print(f'Training: {(size, mode, rho, N, seed)}...')
        qs_with_actions = train_qtables_for_envs(
            envs, qtable_dir,
            agent_name=agent_name,
            show_progress=False,
            just_train_qtables=just_train_qtables,
        )

        if just_train_qtables:
            continue

        qs = [q for (q, _) in qs_with_actions]
        all_action_space = sorted(set().union(*[a for (_, a) in qs_with_actions]))

        # 3. Geometric + coherence measurements (as before)
        geom = measure_one_realisation(envs, qtable_dir, agent_name=agent_name,
                                    run_agent_rollouts=False)

        # 4. (Optional) Agent rollouts
        roll = {}
        if run_rollouts:
            out = run_vm_pms_rollouts(
                envs, qs, all_action_space,
                n_episodes_per_env=1,         # bump to 5+ for variance estimates
                record_per_step=False,
            )
            # Flatten the per-mode summary into roll_<mode>_<metric> columns
            for _, srow in out['summary'].iterrows():
                for col in ['return_mean', 'collisions_mean', 'bps_mean',
                            'success_rate', 'kl_final_mean']:
                    roll[f"{srow['mode']}_{col}"] = srow[col]

        rows.append({
            'size':   f'{size[0]}x{size[1]}',
            'W': size[0], 'H': size[1],
            'mode':   mode,
            'rho':    rho,
            'N':      N,
            'seed':   seed,
            'master_seed': master_seed,
            # The two targeted-experiment-specific columns:
            'F_target': F_target,
            'F_dens_calibrated': F_calib,
            **geom,
            **roll,
        })

        if len(rows) % 50 == 0:
            pd.DataFrame(rows).to_csv(output_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df

# ===================================================================
# 4. Convenience driver: calibration + targeting + sweep in one call
# ===================================================================

def run_targeted_experiment(
    size               = (60, 30),
    rho_grid           = (0.10, 0.20, 0.30),
    F_star_targets     = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40),
    candidate_Ns       = (1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 21),
    chosen_cells       = None,
    max_cells_per_F_star = 4,
    calibration_seeds  = 5,
    n_seeds            = 30,
    mode               = 'iid',
    lambda_cliff       = 100.0,
    grid_family        = 'maze',
    out_dir            = './logs/targeted_sweep/',
    output_csv         = 'targeted_sweep.csv',
    agent_name         = 'qlearning',
    just_train_qtables = False,
):
    """
    Full pipeline: calibrate, pick cells, run targeted sweep.

    Calibration is cheap (no Q-tables) and is always re-run; this
    avoids stale calibrations producing surprising cell selections.
    If you want to cache calibrations, save the returned `calibration`
    DataFrame and pass it back to `pick_targeted_cells` directly.

    Returns
    -------
    dict with keys 'calibration', 'targeted_cells', 'sweep_df'.
    """
    os.makedirs(out_dir, exist_ok=True)

    if chosen_cells is None:
        print("=== Step 1: calibration (no Q-tables) ===")
        calibration = calibrate_N_for_F_star_targets(
            size=size,
            rho_grid=list(rho_grid),
            candidate_Ns=list(candidate_Ns),
            n_seeds=calibration_seeds,
            mode=mode,
            lambda_cliff=lambda_cliff,
            grid_family=grid_family,
        )
        cal_path = os.path.join(out_dir, 'calibration.csv')
        calibration.to_csv(cal_path, index=False)
        print(f"Calibration saved to {cal_path}.  Sample:")
        print(calibration.head(10).to_string(index=False))

        print("\n=== Step 2: pick (rho, N) cells per F*-density target ===")
        targeted_cells = pick_targeted_cells(
            calibration=calibration,
            F_star_targets=list(F_star_targets),
            max_cells_per_F_star=max_cells_per_F_star,
            verbose=True,
        )
    else:
        targeted_cells = pd.DataFrame(chosen_cells, columns=['rho', 'N', 'F_target'])
        targeted_cells['F_dens_calibrated'] = targeted_cells['F_target']
        targeted_cells['miss'] = 0.0
        targeted_cells = targeted_cells.sort_values(['F_target', 'N', 'rho'])
        calibration = None

    tgt_path = os.path.join(out_dir, 'targeted_cells.csv')
    targeted_cells.to_csv(tgt_path, index=False)
    print(f"Targeted cells saved to {tgt_path}:")
    print(targeted_cells.sort_values(['F_target', 'N', 'rho']).to_string(index=False))

    total_qtables = (targeted_cells['N'] * n_seeds).sum()
    print(f"\nMain sweep will train approximately {total_qtables} Q-tables "
          f"({len(targeted_cells)} cells x {n_seeds} seeds, "
          f"mean N = {targeted_cells['N'].mean():.1f}).")

    print("\n=== Step 3: targeted sweep (heavy Q-table training) ===")
    output_csv = os.path.join(out_dir, output_csv)
    qtable_root = os.path.join(out_dir, 'qtables')
    sweep_df = run_targeted_sweep(
        targeted_cells=targeted_cells,
        size=size,
        n_seeds=n_seeds,
        mode=mode,
        output_csv=output_csv,
        qtable_root=qtable_root,
        agent_name=agent_name,
        lambda_cliff=lambda_cliff,
        grid_family=grid_family,
        just_train_qtables=just_train_qtables,
    )

    return {
        'calibration':    calibration,
        'targeted_cells': targeted_cells,
        'sweep_df':       sweep_df,
    }


# ===================================================================
# 5. Analysis: the decisive collapse test
# ===================================================================

"""
analyze_targeted_sweep_patched.py

Drop-in replacement for `analyze_targeted_sweep`.  Adds:

  1. Bin-type-aware thresholds for the verdict logic:
       - 3-rho bins: spread >= 0.20 AND chi-square p < 0.01 -> separation
       - 4-rho bins: spread >= 0.25 AND chi-square p < 0.02 -> separation
       - 3-rho monotonicity: each consecutive bin's p < 0.05
       - 4-rho monotonicity: each consecutive bin's p < 0.10
       - >=3 consecutive monotonic bins triggers "decisive separation"

  2. Monotonicity tally:
       per-bin direction check (voting predicts P(FI) decreases as rho rises);
       counts and prints how many bins show voting-consistent ordering, plus
       null-hypothesis probability.

  3. Nested-bridge verdict:
       takes optional (F_target, F_target_se) for the nested-family-derived
       F*-density target.  Tests whether observed F*_coher lands inside
       the agreement band [F_target +/- 2*se].

  4. Overall verdict synthesis:
       returns one of {'decisive_collapse', 'decisive_separation',
                       'inconclusive', 'insufficient_data'} based on the
       pre-registered criteria.

Usage
-----
    from analyze_targeted_sweep_patched import analyze_targeted_sweep

    summary = analyze_targeted_sweep(
        sweep_df,
        out_dir='./logs/targeted_sweep/',
        nested_bridge_target=(0.692, 0.025),   # F_target, se from your
                                                # nested-bridge calibration
    )
    print(f"Verdict: {summary['verdict']}")
"""

import os
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


# ===================================================================
# Constants
# ===================================================================

P_C_SITE = 0.5927
P_C_BOND = 0.5

# Verdict thresholds (4-rho design, n>=30 per cell).  Bin-type-aware.
COLLAPSE_SPREAD_MAX        = 0.12
COLLAPSE_MIN_P             = 0.10
SEPARATION_3RHO_SPREAD     = 0.20
SEPARATION_3RHO_P          = 0.01
SEPARATION_4RHO_SPREAD     = 0.25
SEPARATION_4RHO_P          = 0.02
MONO_3RHO_P                = 0.05
MONO_4RHO_P                = 0.10
MONO_CONSECUTIVE_REQUIRED  = 3
MIN_N_PER_CELL             = 10   # cells with fewer obs don't count
DEFAULT_NESTED_BRIDGE      = (0.69, 0.05)   # placeholder if user passes None


# ===================================================================
# Per-bin helpers
# ===================================================================

def _classify_bin_monotonicity(rhos, p_fis):
    """
    Classify the rho-ordering of P(FI) values in one F*-bin.

    Returns
    -------
    int
        -1  if P(FI) is non-increasing in rho  (voting-consistent direction:
            higher rho -> tighter voting margin -> more failures -> lower P(FI))
        +1  if P(FI) is non-decreasing in rho  (anti-voting direction)
         0  if non-monotonic
        +2  if all values are identical (ties; counts as neutral evidence)
    """
    pairs = sorted(zip(rhos, p_fis))
    p_sorted = [p for _, p in pairs]
    if len(set(p_sorted)) == 1:
        return +2
    is_nonincreasing = all(p_sorted[i] >= p_sorted[i+1]
                           for i in range(len(p_sorted)-1))
    is_nondecreasing = all(p_sorted[i] <= p_sorted[i+1]
                           for i in range(len(p_sorted)-1))
    if is_nonincreasing and not is_nondecreasing:
        return -1
    if is_nondecreasing and not is_nonincreasing:
        return +1
    return 0


def _null_prob_monotonic(n_rhos):
    """
    Under H0 (P(FI) is independent of rho conditional on F*-density),
    the probability that all n_rhos sample proportions happen to be
    monotonically ordered in either direction is 2 / n_rhos!.

    For n=3: 2/6   = 0.333
    For n=4: 2/24  = 0.083
    For n=5: 2/120 = 0.017
    """
    if n_rhos < 2:
        return 1.0
    return 2.0 / math.factorial(n_rhos)


def _max_consecutive_monotonic(bin_rows, mono_direction=-1):
    """
    Find the longest run of consecutive F*-bins (sorted by F*-density)
    whose monotonicity classification equals `mono_direction` AND whose
    chi-square p-value is below the bin-type-specific threshold.

    Parameters
    ----------
    bin_rows : list of dicts
        Each dict has keys: 'F_center', 'n_rhos', 'mono_class',
        'chi2_p', 'p_thresh'.  Pre-sorted by F_center.
    mono_direction : -1 or +1
        Which direction counts.  -1 = voting-consistent (P(FI) decreasing
        in rho); +1 = anti-voting.

    Returns
    -------
    int : length of longest run.
    """
    longest = 0
    current = 0
    for r in bin_rows:
        if (r['mono_class'] == mono_direction
                and r['chi2_p'] < r['p_thresh']):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


# ===================================================================
# Main analyzer
# ===================================================================

def analyze_targeted_sweep(
    sweep_df,
    out_dir              = './logs/targeted_sweep/',
    F_bin_width          = 0.05,
    min_n_per_cell       = MIN_N_PER_CELL,
    nested_bridge_target = None,
    make_plots           = True,
):
    """
    Patched analysis: monotonicity tally + bin-type-aware verdict.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of run_targeted_sweep.  Required columns: rho, N, seed,
        F_star_density, C_interp.
    out_dir : str
        Where to write plots and CSV outputs.
    F_bin_width : float
        Width of F*-density bins for grouping.
    min_n_per_cell : int
        Bins with fewer obs per (F_bin, rho) cell than this don't
        contribute to the verdict.
    nested_bridge_target : tuple (F_target, se) or None
        Measured nested-family-derived F*-density target with standard
        error.  If None, uses DEFAULT_NESTED_BRIDGE as placeholder.
    make_plots : bool
        If True, write the collapse and failure-mode plots.

    Returns
    -------
    dict with keys:
        verdict           : str in {'decisive_collapse', 'decisive_separation',
                                    'inconclusive', 'insufficient_data'}
        verdict_reasons   : list of strings explaining the verdict
        F_star_coher_50pct: estimated F*-density at 50% coherence failure
        F_star_topo_50pct : estimated F*-density at 50% topology failure
        bridge_target     : (F_target, se) used
        bridge_pass       : bool, whether observed F*_coher is inside the band
        monotonic_tally   : dict summarising the monotonicity test
        bin_table         : pd.DataFrame, per-bin test results
        plots             : list of plot paths (if make_plots)
    """
    import matplotlib
    if make_plots:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    os.makedirs(out_dir, exist_ok=True)

    if nested_bridge_target is None:
        nested_bridge_target = DEFAULT_NESTED_BRIDGE
    F_bridge, se_bridge = nested_bridge_target

    df = sweep_df.copy()
    df['fi_holds']   = df['C_interp'].isin(['marginal', 'free'])
    df['fail_coher'] = df['C_interp'] == 'incoherent'
    df['fail_topo']  = df['C_interp'] == 'no_path'

    # Bin REALIZED F*-density (not target).  This is the key distinction
    # from the parametric-sweep analysis.
    bins = np.arange(0.30, 1.01, F_bin_width)
    df['F_bin'] = pd.cut(df['F_star_density'], bins=bins, include_lowest=True)

    rhos = sorted(df['rho'].unique())

    # -----------------------------------------------------------------
    # Per-bin / per-rho aggregation
    # -----------------------------------------------------------------
    per_cell = (df.groupby(['F_bin', 'rho'], observed=True)
                  .agg(n=('fi_holds', 'count'),
                       successes=('fi_holds', 'sum'),
                       p_fi=('fi_holds', 'mean'),
                       p_coher=('fail_coher', 'mean'),
                       p_topo=('fail_topo', 'mean'),
                       F_realized=('F_star_density', 'mean'))
                  .reset_index())
    per_cell = per_cell[per_cell['n'] >= min_n_per_cell]

    # -----------------------------------------------------------------
    # Per-bin: chi-square, monotonicity, spread
    # -----------------------------------------------------------------
    bin_rows = []
    for fb, sub in per_cell.groupby('F_bin', observed=True):
        n_rhos = len(sub)
        if n_rhos < 2:
            continue

        # Chi-square test on the rho x outcome contingency table
        contingency = np.vstack([
            sub['successes'].values,
            (sub['n'] - sub['successes']).values,
        ])
        if (contingency.sum(axis=1) == 0).any():
            chi2, pval, dof = float('nan'), float('nan'), n_rhos - 1
        else:
            try:
                chi2, pval, dof, _ = stats.chi2_contingency(contingency)
            except ValueError:
                chi2, pval = float('nan'), float('nan')
                dof = n_rhos - 1

        spread = float(sub['p_fi'].max() - sub['p_fi'].min())
        mono_class = _classify_bin_monotonicity(
            sub['rho'].values, sub['p_fi'].values)

        # Bin-type-aware separation thresholds
        if n_rhos == 3:
            sep_spread = SEPARATION_3RHO_SPREAD
            sep_p      = SEPARATION_3RHO_P
            mono_p     = MONO_3RHO_P
        elif n_rhos >= 4:
            sep_spread = SEPARATION_4RHO_SPREAD
            sep_p      = SEPARATION_4RHO_P
            mono_p     = MONO_4RHO_P
        else:
            sep_spread = float('inf')
            sep_p      = 0.0
            mono_p     = 0.0

        bin_rows.append({
            'F_bin':       fb,
            'F_center':    float(fb.mid),
            'n_rhos':      n_rhos,
            'rhos':        list(sub['rho']),
            'p_fi':        [float(x) for x in sub['p_fi']],
            'ns':          [int(x)   for x in sub['n']],
            'spread':      spread,
            'chi2':        float(chi2),
            'chi2_p':      float(pval),
            'mono_class':  mono_class,
            'sep_spread':  sep_spread,
            'sep_p':       sep_p,
            'p_thresh':    mono_p,
            'separates':   (spread >= sep_spread and pval < sep_p)
                            if not math.isnan(pval) else False,
        })

    if len(bin_rows) == 0:
        return None
    
    bin_table = pd.DataFrame(bin_rows).sort_values('F_center').reset_index(drop=True)

    # Write detailed per-bin table to CSV
    bin_table_csv = os.path.join(out_dir, 'cross_rho_tests.csv')
    bin_table.to_csv(bin_table_csv, index=False)

    # -----------------------------------------------------------------
    # Monotonicity tally
    # -----------------------------------------------------------------
    voting_consistent_bins = bin_table[bin_table['mono_class'] == -1]
    anti_voting_bins       = bin_table[bin_table['mono_class'] == +1]
    nonmono_bins           = bin_table[bin_table['mono_class'] == 0]
    tied_bins              = bin_table[bin_table['mono_class'] == +2]

    # Expected null-hypothesis count of voting-consistent monotonic bins:
    # for each bin, P(monotonic in voting direction) = 1/n_rhos!.
    n_qualifying = len(bin_table)
    expected_null_mono = sum(1.0 / math.factorial(r['n_rhos'])
                             for _, r in bin_table.iterrows())
    observed_mono = len(voting_consistent_bins)

    # One-sided binomial-ish test: how surprising is observing this many
    # voting-consistent bins?  Use a simple binomial approximation with
    # mean probability of voting-consistent monotonic ordering averaged
    # over bins.
    if n_qualifying > 0:
        avg_null_p = expected_null_mono / n_qualifying
        # P(X >= observed_mono | X ~ Binomial(n_qualifying, avg_null_p))
        binom_p = float(1 - stats.binom.cdf(observed_mono - 1,
                                            n_qualifying, avg_null_p))
    else:
        binom_p = float('nan')

    longest_voting_run = _max_consecutive_monotonic(
        bin_table.to_dict('records'), mono_direction=-1)
    longest_anti_run = _max_consecutive_monotonic(
        bin_table.to_dict('records'), mono_direction=+1)

    monotonic_tally = {
        'n_qualifying_bins':     n_qualifying,
        'voting_consistent':     observed_mono,
        'anti_voting':           len(anti_voting_bins),
        'non_monotonic':         len(nonmono_bins),
        'tied':                  len(tied_bins),
        'expected_null_count':   expected_null_mono,
        'binom_p_observed':      binom_p,
        'longest_voting_run':    longest_voting_run,
        'longest_anti_run':      longest_anti_run,
    }

    # -----------------------------------------------------------------
    # F*_coher and F*_topo: 50% threshold estimates
    #
    # IMPORTANT: P(coher fail) is non-monotonic in F*-density.  It rises
    # from 0 as F* falls from 1.0 (coherence becomes the binding
    # mechanism), reaches a maximum, then falls as topology takes over
    # and crowds out coherence as the dominant failure mode.  We want
    # the UPPER edge: the highest F*-density at which P(coher) >= 0.5,
    # going downward from F* = 1.0.  This is the "coherence onset"
    # threshold and the one that matches the nested-bridge target.
    #
    # P(topo fail) is monotonic in F*-density (rises as F* falls), so
    # for it either edge gives the same answer.  We use the same
    # "downward search" for consistency.
    # -----------------------------------------------------------------
    overall = (df.groupby('F_bin', observed=True)
                 .agg(p_coher=('fail_coher', 'mean'),
                      p_topo=('fail_topo', 'mean'),
                      p_fi=('fi_holds', 'mean'),
                      f=('F_star_density', 'mean'),
                      n=('fi_holds', 'count'))
                 .reset_index())
    overall = overall[overall['n'] >= min_n_per_cell].sort_values('f', ascending=False)

    def threshold_descending(p_series, f_series, level=0.5):
        """
        Find the highest F*-density at which `p_series` first crosses
        `level` going DOWNWARD from F* = 1.0.

        Both series are passed in F*-descending order.

        Returns
        -------
        (F_threshold, is_lower_bound) : tuple
            F_threshold: the estimated F*-density at the crossing.
            is_lower_bound: True if the highest-F* bin we observed is
                already above the level (meaning the true threshold
                lies at or above the data range we measured); False if
                a clean crossing was observed inside the data range.
            Both are NaN/False when no crossing is observed.
        """
        p_arr = p_series.values
        f_arr = f_series.values
        for i in range(len(p_arr) - 1):
            # Going from high F* (where P is below threshold) to lower F*
            # (where P crosses above threshold).
            if p_arr[i] < level and p_arr[i+1] >= level:
                return (float((f_arr[i] + f_arr[i+1]) / 2), False)
        # If the FIRST bin (highest F*) is already above level, the
        # threshold is at or above the highest observed F* -- this is
        # a LOWER BOUND, not a point estimate.
        if len(p_arr) > 0 and p_arr[0] >= level:
            return (float(f_arr[0]), True)
        return (float('nan'), False)

    F_coher_50, F_coher_50_lb_flag = threshold_descending(
        overall['p_coher'], overall['f'])
    F_topo_50,  F_topo_50_lb_flag  = threshold_descending(
        overall['p_topo'],  overall['f'])

    # -----------------------------------------------------------------
    # Multi-level FI-threshold table
    #
    # Reports the F*-density at which P(FI) first drops below each of
    # 0.9, 0.5, and 0.1 as F* decreases from 1.0.  These correspond to
    # different operational definitions of "the FI threshold":
    #   0.9: FI starts to break down (1-in-10 failure rate; matches the
    #        nested-family rho_c definition of "first non-trivial
    #        failure rate")
    #   0.5: median FI threshold (50/50 success)
    #   0.1: FI has substantially collapsed (90% failure rate)
    #
    # The nested-bridge target may correspond to any of these depending
    # on how the nested rho_c was defined.  Comparing all three gives
    # the reader the right cross-protocol comparison without committing
    # to a specific level a priori.
    # -----------------------------------------------------------------
    fi_levels = [0.9, 0.5, 0.1]
    fi_thresholds = {}
    for level in fi_levels:
        # P(FI) descends as F* descends, so we want the highest F* at
        # which P(FI) < level.  Equivalently: P(failure) >= 1-level.
        # We invert via the descending search on (1 - p_fi).
        threshold_p_fail = 1.0 - level
        thr, lb = threshold_descending(
            1 - overall['p_fi'], overall['f'], level=threshold_p_fail)
        # NB: thr is the F* where (1-p_fi) first crosses threshold_p_fail
        # going downward.  Since p_fi DECREASES as F* decreases, (1-p_fi)
        # INCREASES as F* decreases, so the crossing is real and
        # downward.  The result is the F* where p_fi first drops below
        # `level`.
        fi_thresholds[level] = {
            'F_threshold':   thr,
            'is_lower_bound': lb,
        }

    # Bridge test using P(FI)=0.9 by default (the "FI starts to fail"
    # definition that best matches the nested rho_c interpretation).
    # User can compare to other levels in the printed output.
    bridge_band = (F_bridge - 2*se_bridge, F_bridge + 2*se_bridge)
    bridge_disagree_band = (F_bridge - 4*se_bridge, F_bridge + 4*se_bridge)
    F_bridge_compare = fi_thresholds[0.9]['F_threshold']
    F_bridge_is_lb   = fi_thresholds[0.9]['is_lower_bound']

    if not math.isnan(F_bridge_compare):
        if F_bridge_is_lb:
            # We only have a lower bound; bridge test is inconclusive on
            # the band-membership question.  We can only say "F* >= ...".
            # Treat as PASS only if the lower bound is BELOW the upper
            # edge of the agreement band -- because the true value
            # could be anywhere above the lower bound.
            bridge_pass = (F_bridge_compare <= bridge_band[1])
            bridge_disagree = (F_bridge_compare > bridge_disagree_band[1])
        else:
            bridge_pass = (bridge_band[0] <= F_bridge_compare <= bridge_band[1])
            bridge_disagree = not (bridge_disagree_band[0] <= F_bridge_compare
                                   <= bridge_disagree_band[1])
    else:
        bridge_pass = False
        bridge_disagree = False

    # -----------------------------------------------------------------
    # Verdict synthesis
    # -----------------------------------------------------------------
    verdict = None
    reasons = []

    if n_qualifying == 0:
        verdict = 'insufficient_data'
        reasons.append("No bins have >=2 rho values with sufficient sample "
                       f"size (min_n_per_cell={min_n_per_cell}).")
    else:
        # Check decisive separation triggers (any one suffices)
        separating_bins = bin_table[bin_table['separates']]
        mono_run_triggered = longest_voting_run >= MONO_CONSECUTIVE_REQUIRED

        if len(separating_bins) > 0:
            verdict = 'decisive_separation'
            for _, r in separating_bins.iterrows():
                reasons.append(
                    f"Bin {r['F_bin']} ({r['n_rhos']}-rho): spread="
                    f"{r['spread']:.3f} >= {r['sep_spread']:.2f}, "
                    f"chi2 p={r['chi2_p']:.4f} < {r['sep_p']:.3f}.")
        elif mono_run_triggered:
            verdict = 'decisive_separation'
            reasons.append(
                f"Voting-consistent monotonic ordering observed across "
                f"{longest_voting_run} consecutive F*-bins, each with "
                f"chi-square p below its bin-type threshold.")
        elif bridge_disagree:
            verdict = 'decisive_separation'
            reasons.append(
                f"F* threshold for P(FI)=0.9 = "
                f"{'>=' if F_bridge_is_lb else ''}{F_bridge_compare:.3f} "
                f"lies outside the nested-bridge 4*se disagreement band "
                f"({bridge_disagree_band[0]:.3f}, "
                f"{bridge_disagree_band[1]:.3f}).")
        else:
            # No separation triggered; check decisive collapse criteria
            max_spread = bin_table['spread'].max()
            min_p = bin_table['chi2_p'].min()
            collapse_a = max_spread < COLLAPSE_SPREAD_MAX
            collapse_b = min_p > COLLAPSE_MIN_P
            collapse_c = bridge_pass and not F_bridge_is_lb  # need a point estimate, not a bound

            if collapse_a and collapse_b and collapse_c:
                verdict = 'decisive_collapse'
                reasons.append(
                    f"Max spread {max_spread:.3f} < {COLLAPSE_SPREAD_MAX}.")
                reasons.append(
                    f"Min chi-square p {min_p:.4f} > {COLLAPSE_MIN_P}.")
                reasons.append(
                    f"F* at P(FI)=0.9 = {F_bridge_compare:.3f} lies inside "
                    f"nested-bridge agreement band "
                    f"[{bridge_band[0]:.3f}, {bridge_band[1]:.3f}].")
            else:
                verdict = 'inconclusive'
                if not collapse_a:
                    reasons.append(
                        f"Max spread {max_spread:.3f} exceeds collapse "
                        f"threshold ({COLLAPSE_SPREAD_MAX}).")
                if not collapse_b:
                    reasons.append(
                        f"Min chi-square p {min_p:.4f} below collapse "
                        f"threshold ({COLLAPSE_MIN_P}).")
                if F_bridge_is_lb:
                    reasons.append(
                        f"F* at P(FI)=0.9 is only known as a lower bound "
                        f"(>={F_bridge_compare:.3f}); the true value lies "
                        f"above the data range and the bridge test is "
                        f"inconclusive.  Extend the F*-target grid to "
                        f"higher F*-density to resolve.")
                elif not collapse_c:
                    reasons.append(
                        f"F* at P(FI)=0.9 = {F_bridge_compare:.3f} outside "
                        f"nested-bridge agreement band "
                        f"[{bridge_band[0]:.3f}, {bridge_band[1]:.3f}].")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    plot_paths = []
    if make_plots:
        colours = plt.cm.viridis(np.linspace(0.10, 0.90, len(rhos)))
        rho_colour = dict(zip(rhos, colours))

        # Plot 1: collapse test with nested-bridge band overlaid
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for rho in rhos:
            sr = df[df['rho'] == rho]
            g = (sr.groupby('F_bin', observed=True)
                    .agg(p_fi=('fi_holds', 'mean'),
                         f=('F_star_density', 'mean'),
                         n=('fi_holds', 'count'))
                    .reset_index())
            g = g[g['n'] >= 3]
            g['se'] = (g['p_fi'] * (1 - g['p_fi']) / g['n']).pow(0.5)
            ax.errorbar(g['f'], g['p_fi'], yerr=g['se'],
                        marker='o', label=f'rho={rho:.2f}',
                        color=rho_colour[rho], capsize=3, alpha=0.85)
        ax.axvspan(bridge_band[0], bridge_band[1], alpha=0.15, color='green',
                   label=f'nested bridge band')
        ax.axvline(F_bridge, color='green', linestyle='--', alpha=0.6,
                   label=f'$F^*_{{nested}}$={F_bridge:.3f}')
        ax.axvline(P_C_SITE, color='red',  linestyle=':',  label=r'$p_c^{site}$')
        ax.axvline(P_C_BOND, color='red',  linestyle='-.', label=r'$p_c^{bond}$',
                   alpha=0.5)
        # Mark FI-level threshold crossings as vertical dotted lines
        level_colours = {0.9: '#1f77b4', 0.5: '#7f7f7f', 0.1: '#9467bd'}
        for level in fi_levels:
            info = fi_thresholds[level]
            if not math.isnan(info['F_threshold']):
                lb_tag = '>=' if info['is_lower_bound'] else ''
                ax.axvline(info['F_threshold'], color=level_colours[level],
                           linestyle=':', alpha=0.7,
                           label=f'P(FI)={level}: '
                                 f'F*{lb_tag}{info["F_threshold"]:.3f}')
                # Horizontal reference line at the corresponding P(FI) level
                ax.axhline(level, color=level_colours[level], linestyle=':',
                           alpha=0.25)
        ax.set_xlabel(r'realized $|F^*| / |\mathcal{S}|$')
        ax.set_ylabel('P(FI holds)')
        ax.set_title(f'Targeted sweep: collapse test  (verdict: {verdict})')
        ax.set_xlim(0.30, 1.00)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='center right')
        fig.tight_layout()
        p1 = os.path.join(out_dir, 'plot1_collapse.png')
        fig.savefig(p1, dpi=120)
        plt.close(fig)
        plot_paths.append(p1)

        # Plot 2: failure-mode decomposition per rho
        if len(rhos) > 0:
            fig, axes = plt.subplots(1, len(rhos),
                                     figsize=(3.8 * len(rhos), 3.5),
                                     sharey=True, squeeze=False)
            for ax, rho in zip(axes[0], rhos):
                sr = df[df['rho'] == rho]
                g = (sr.groupby('F_bin', observed=True)
                        .agg(p_coher=('fail_coher', 'mean'),
                             p_topo=('fail_topo', 'mean'),
                             f=('F_star_density', 'mean'),
                             n=('fi_holds', 'count'))
                        .reset_index())
                g = g[g['n'] >= 3].sort_values('f')
                ax.fill_between(g['f'], 0, g['p_topo'],
                                color='#d62728', alpha=0.6,
                                label='P(topo fail)')
                ax.fill_between(g['f'], g['p_topo'],
                                g['p_topo'] + g['p_coher'],
                                color='#ff7f0e', alpha=0.6,
                                label='P(coher fail)')
                ax.fill_between(g['f'], g['p_topo'] + g['p_coher'], 1.0,
                                color='#2ca02c', alpha=0.6,
                                label='P(FI holds)')
                ax.axvline(P_C_SITE, color='black', linestyle=':')
                ax.axvline(P_C_BOND, color='black', linestyle='-.', alpha=0.5)
                ax.axvline(F_bridge, color='green', linestyle='--', alpha=0.6)
                ax.set_xlabel(r'$|F^*| / |\mathcal{S}|$')
                ax.set_title(f'rho={rho:.2f}')
                ax.set_xlim(0.30, 1.00)
                ax.set_ylim(0, 1)
            axes[0][0].set_ylabel('Probability')
            axes[0][-1].legend(fontsize=7, loc='center left')
            fig.suptitle('Failure-mode decomposition per rho')
            fig.tight_layout()
            p2 = os.path.join(out_dir, 'plot2_failure_modes.png')
            fig.savefig(p2, dpi=120)
            plt.close(fig)
            plot_paths.append(p2)

        # Plot 3: monotonicity tally visual
        # For each bin, mark whether it's voting-consistent, anti-voting,
        # non-monotonic, or tied.
        if len(bin_table) > 0:
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ymap = {-1: 'voting-consistent', +1: 'anti-voting',
                     0: 'non-monotonic',     +2: 'tied'}
            colour_map = {-1: '#2ca02c', +1: '#d62728',
                           0: '#1f77b4', +2: '#7f7f7f'}
            for _, r in bin_table.iterrows():
                ax.scatter(r['F_center'], 0,
                           color=colour_map[r['mono_class']],
                           s=160 if r['n_rhos'] >= 4 else 100,
                           edgecolor='black', linewidth=0.5,
                           zorder=3)
                ax.annotate(f"{r['n_rhos']}",
                            (r['F_center'], 0),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=7)
            ax.axvline(F_bridge, color='green', linestyle='--', alpha=0.6,
                       label=f'$F^*_{{nested}}$')
            ax.axvline(P_C_SITE, color='red', linestyle=':', label=r'$p_c^{site}$')
            ax.set_xlim(0.30, 1.00)
            ax.set_yticks([])
            ax.set_xlabel(r'$|F^*| / |\mathcal{S}|$')
            ax.set_title(f"Monotonicity per bin "
                         f"(voting-consistent: {observed_mono}/{n_qualifying}, "
                         f"longest run: {longest_voting_run})")
            handles = [plt.scatter([], [], color=colour_map[k], s=80,
                                   edgecolor='black', linewidth=0.5, label=v)
                       for k, v in ymap.items()]
            ax.legend(handles=handles, fontsize=7, loc='upper right')
            fig.tight_layout()
            p3 = os.path.join(out_dir, 'plot3_monotonicity.png')
            fig.savefig(p3, dpi=120)
            plt.close(fig)
            plot_paths.append(p3)

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------
    def fmt_threshold(F_val, is_lb):
        """Format a threshold, marking lower-bound values explicitly."""
        if math.isnan(F_val):
            return 'NaN (no crossing observed)'
        if is_lb:
            return f'>= {F_val:.3f}  (lower bound; true value above data range)'
        return f'{F_val:.3f}'

    print()
    print("=" * 72)
    print("TARGETED SWEEP ANALYSIS")
    print("=" * 72)
    print(f"Bins analysed:         {n_qualifying}")
    print()
    print("--- Failure-mechanism thresholds ---")
    print(f"  F*_coher (P(coher)=0.5 upper edge): "
          f"{fmt_threshold(F_coher_50, F_coher_50_lb_flag)}")
    print(f"  F*_topo  (P(topo)=0.5 transition):  "
          f"{fmt_threshold(F_topo_50,  F_topo_50_lb_flag)}")
    print()
    print("--- Multi-level FI thresholds (cross-protocol comparison) ---")
    print(f"  F*-density at which P(FI) drops below each level,")
    print(f"  going downward from F* = 1.0:")
    print(f"     {'level':>10}  {'F*-density':>32}  notes")
    for level in fi_levels:
        info = fi_thresholds[level]
        thr_str = fmt_threshold(info['F_threshold'], info['is_lower_bound'])
        if level == 0.9:
            note = "(FI starts to fail; closest to nested rho_c)"
        elif level == 0.5:
            note = "(median; 50/50 success)"
        elif level == 0.1:
            note = "(deep failure; bottom of FI regime)"
        else:
            note = ""
        print(f"     {level:>10.1f}  {thr_str:>32}  {note}")
    print()
    print("--- Cross-protocol bridge test ---")
    print(f"  Nested-bridge target:  {F_bridge:.3f} +/- {se_bridge:.3f}")
    print(f"  Agreement band 2*se:   [{bridge_band[0]:.3f}, "
          f"{bridge_band[1]:.3f}]")
    print(f"  Disagreement band 4*se: [{bridge_disagree_band[0]:.3f}, "
          f"{bridge_disagree_band[1]:.3f}]")
    print(f"  Using F*-density at P(FI) = 0.9 for the comparison: "
          f"{fmt_threshold(F_bridge_compare, F_bridge_is_lb)}")
    print(f"  Bridge test:           "
          f"{'PASS' if bridge_pass else ('INCONCLUSIVE (lb only)' if F_bridge_is_lb else 'FAIL')}")
    print()
    print("--- Monotonicity tally ---")
    print(f"  Voting-consistent (P(FI) falling in rho): "
          f"{observed_mono}/{n_qualifying}")
    print(f"  Anti-voting  (P(FI) rising in rho):      "
          f"{len(anti_voting_bins)}/{n_qualifying}")
    print(f"  Non-monotonic:                            "
          f"{len(nonmono_bins)}/{n_qualifying}")
    print(f"  Tied (all P(FI) equal):                   "
          f"{len(tied_bins)}/{n_qualifying}")
    print(f"  Expected voting-consistent under null:    "
          f"{expected_null_mono:.2f}")
    print(f"  Binomial p (observed >= expected):        "
          f"{binom_p:.4f}")
    print(f"  Longest voting-consistent run (sig.):     "
          f"{longest_voting_run}")
    print(f"  Longest anti-voting run (sig.):           "
          f"{longest_anti_run}")
    print()
    print("--- Per-bin chi-square tests ---")
    print(f"  Max spread across bins:           "
          f"{bin_table['spread'].max():.3f}")
    print(f"  Min chi-square p across bins:     "
          f"{bin_table['chi2_p'].min():.4f}")
    print(f"  Bins triggering separation rule:  "
          f"{int(bin_table['separates'].sum())}")
    print()
    print(f"--- VERDICT: {verdict.upper()} ---")
    for r in reasons:
        print(f"  - {r}")
    print()
    if plot_paths:
        print("Plots:")
        for p in plot_paths:
            print(f"  {p}")
    print("=" * 72)

    return {
        'verdict':             verdict,
        'verdict_reasons':     reasons,
        'F_star_coher_50pct':  F_coher_50,
        'F_star_coher_is_lb':  F_coher_50_lb_flag,
        'F_star_topo_50pct':   F_topo_50,
        'F_star_topo_is_lb':   F_topo_50_lb_flag,
        'fi_thresholds':       fi_thresholds,    # {level: {F_threshold, is_lower_bound}}
        'bridge_target':       nested_bridge_target,
        'bridge_compare_F':    F_bridge_compare,
        'bridge_compare_is_lb': F_bridge_is_lb,
        'bridge_pass':         bridge_pass,
        'monotonic_tally':     monotonic_tally,
        'bin_table':           bin_table,
        'plots':               plot_paths,
    }

# ---------------------------------------------------------------
# 4. Aggregation / plots.
# ---------------------------------------------------------------
def aggregate_success_rate(df):
    """
    Plot A: P(FI holds) as a function of (rho, N), one panel per (size, mode).

    'FI holds' is defined as C_V >= 0 AND C_interp != 'no_path',
    i.e. either 'marginal' or 'free' or 'incoherent-but-not-disconnected'
    -- adjust to taste.  For the discriminator I recommend the strict:
        success := (C_interp in {'marginal', 'free'})
    so 'no_path' (Mechanism I) and 'incoherent' (Mechanism II) both count
    as failure but are *distinguishable* in the auxiliary plots.
    """
    df = df.copy()
    df['fi_holds']      = df['C_interp'].isin(['marginal', 'free'])
    df['fail_topo']     = df['C_interp'] == 'no_path'
    df['fail_coher']    = df['C_interp'] == 'incoherent'

    grouped = (df
               .groupby(['size', 'mode', 'rho', 'N'])
               .agg(
                    p_fi_holds   = ('fi_holds',   'mean'),
                    p_fail_topo  = ('fail_topo',  'mean'),
                    p_fail_coher = ('fail_coher', 'mean'),
                    F_dens_mean  = ('F_star_density', 'mean'),
                    F_dens_std   = ('F_star_density', 'std'),
                    n_seeds      = ('seed',       'count'),
               )
               .reset_index())
    return grouped


def plot_A_success_vs_N(grouped, size_filter='100x50', fig_name=None):
    """One curve per rho; x = N, y = P(FI holds)."""
    import matplotlib.pyplot as plt
    sub = grouped[grouped['size'] == size_filter]
    modes = sub['mode'].value_counts().index.to_list()
    n_modes = len(modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(5*n_modes+1, 4), sharey=True)
    axes = axes  if n_modes > 1 else [axes]
    for ax, mode in zip(axes, modes):
        s = sub[sub['mode'] == mode]
        for rho, g in s.groupby('rho'):
            g = g.sort_values('N')
            ax.plot(g['N'], g['p_fi_holds'], 'o-', label=f'rho={rho:.2f}')
            # overlay percolation prediction
            N_star = np.log(0.5927) / np.log(1 - rho)
            ax.axvline(N_star, color='gray', ls=':', alpha=0.4)
        ax.set_xlabel('N'); ax.set_title(f'{mode}, {size_filter}')
        ax.set_xscale('log'); ax.legend()
    axes[0].set_ylabel('P(FI holds)')
    fig.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name, dpi=120, bbox_inches='tight')        
    return fig


def plot_B_collapse(df, size_filter='100x50', fig_name=None):
    """
    P(FI holds) vs F*-density.  If percolation is the mechanism,
    points should collapse onto a single sigmoid centred near
    p_c_site ~= 0.5927, regardless of (rho, N).
    """
    import matplotlib.pyplot as plt
    sub = df[df['size'] == size_filter].copy()
    sub['fi_holds'] = sub['C_interp'].isin(['marginal', 'free'])

    # bin by F*-density
    bins = np.linspace(0.0, 1.0, 41)
    sub['F_bin'] = pd.cut(sub['F_star_density'], bins)
    g = sub.groupby(['mode', 'F_bin'], observed=False)['fi_holds'].agg(['mean', 'count']).reset_index()
    g['F_centre'] = g['F_bin'].apply(lambda b: b.mid)

    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, gm in g.groupby('mode'):
        ax.plot(gm['F_centre'], gm['mean'], 'o-', label=mode)
    ax.axvline(0.5927, color='r', ls='--', label='$p_c^{site}$')
    ax.set_xlabel(r'$|F^*| / |\mathcal{S}|$')
    ax.set_ylabel('P(FI holds)')
    ax.legend(); fig.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name, dpi=120, bbox_inches='tight')          
    return fig


if __name__ == "__main__":

    sizes = [(10, 5), (20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    sizes = [(20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    sizes = [(40, 20), (60,30), (80, 40), (100, 50)]
    sizes = [(60,30), (80, 40), (100, 50)]
    sizes = [(80, 40), (100, 50)]
    sizes = [(100, 50)]
    # sizes = [(10, 5)]
    # sizes = [(20, 10)]
    sizes = [(40, 20)]
    sizes = [(60,30)]
    # sizes = [(80, 40)]
    # sizes = [(100, 50)]    

    # Sanity test!!!
    envs = build_fixed_rho_family(size=max(sizes), rho=0.3, N=5,
                                mode='iid', master_seed=12345)[0]
    for i, e in enumerate(envs):
        walls = sorted(getattr(e, 'walls', getattr(e, 'obstacles', set())))
        print(f'env {i}: name={e.name}  |walls|={len(walls)}  hash={hash(tuple(walls)) & 0xFFFF:04x}')

    print()

    use_nested_grids = True # False # 
    run_dense_experiment = True # False # 
    just_train_qtables = False # True # 

    if use_nested_grids:

        test_name = 'targeted_nested_sweep'

        # Manually specify the (rho_max, N) cells.  These span the bridge
        # region with N-variation at each F*-target.

        full_nested_cells = [
            # (rho_max, N)  -- predicted phi ~ 0.785
            (0.50, 2, 0.785), (0.50, 5, 0.785), (0.50, 10, 0.785),
            # phi ~ 0.742
            (0.60, 2, 0.742), (0.60, 5, 0.742), (0.60, 10, 0.742),
            # phi ~ 0.699 (the nested-bridge target itself)
            (0.70, 2, 0.699), (0.70, 5, 0.699), (0.70, 10, 0.699),
            # phi ~ 0.656
            (0.80, 2, 0.656), (0.80, 5, 0.656), (0.80, 10, 0.656),
            # phi ~ 0.613
            (0.90, 2, 0.613), (0.90, 5, 0.613), (0.90, 10, 0.613),
        ]

        # Loop over cells, train Q-tables, score FI per seed.  This is the
        # "manual run" path that bypasses the targeted-sweep harness, going
        # straight through build_fixed_rho_family + train_qtables + score_FI.

        fast_nested_cells = [
            # The bridge target itself, three N values to test the N-effect prediction
            (0.70, 2, 0.699),    # predicts P(FI) ~ 0.20 (the regression extrapolation)
            (0.70, 5, 0.699),    # predicts P(FI) ~ 0.04
            (0.70, 10, 0.699),   # predicts P(FI) ~ 0.0005
            
            # A higher-rho cell to check that the nested protocol agrees
            # with the non-nested protocol at the deep-failure threshold
            (0.80, 5, 0.656),    # phi ~ 0.66; should give P(FI) very close to 0
            
            # A lower-rho cell to anchor the bridge agreement in the
            # high-P(FI) regime
            (0.60, 2, 0.742),    # phi ~ 0.74; should give P(FI) very close to 1
            (0.60, 5, 0.742),    # phi ~ 0.74; predicts P(FI) ~ 0.85 if regression holds
            (0.60, 10, 0.742),   # phi ~ 0.74; predicts P(FI) ~ 0.3
        ]        

        chosen_cells = fast_nested_cells
        rho_grid           = (0.60, 0.70, 0.80)
        F_star_targets     = (0.656, 0.699, 0.742)
        candidate_Ns       = (2, 5, 10)
        max_cells_per_F_star = 3
        n_seeds = 50         
        mode               = 'nested'
    else:
        chosen_cells = None
        if run_dense_experiment:
            test_name = 'targeted_dense_sweep'

            rho_grid           = (0.10, 0.15, 0.18, 0.20, 0.23, 0.25, 0.28, 0.30, 0.33, 0.35)
            F_star_targets     = (0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85)
            candidate_Ns       = (2, 3, 4, 5)
            max_cells_per_F_star = 3
            n_seeds = 50 
            mode               = 'iid'
        else:
            test_name = 'targeted_sweep'

            rho_grid           = (0.10, 0.20, 0.30, 0.40)
            F_star_targets     = (0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40)
            candidate_Ns       = (2, 3, 4, 5, 6, 8, 10, 13, 17, 21)
            max_cells_per_F_star = 5
            n_seeds = 30
            mode               = 'iid'

    min_n_per_cell = n_seeds // 3

    out_dir = f'./logs/{test_name}-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_csv     = f'{test_name}-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}.csv'

    if not os.path.exists(output_csv):

        result = run_targeted_experiment(
            size               = sizes[0], # (60, 30),
            rho_grid           =  rho_grid,
            F_star_targets     = F_star_targets,
            candidate_Ns       = candidate_Ns,
            chosen_cells       = chosen_cells,
            max_cells_per_F_star = max_cells_per_F_star,
            calibration_seeds  = 5,
            n_seeds            = n_seeds, # 5, # 30,
            mode               = mode,
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
    