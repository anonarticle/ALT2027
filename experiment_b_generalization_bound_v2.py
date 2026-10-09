"""
Experiment (b) — corrected version.

Tests the generalization bound (Theorem 7.2) by sampling random entry-level
assignments ℓ : C → {1, ..., N}, evaluating ε_margin under the FIXED family's
Q_mix on each, and comparing the test-set failure rate to the predicted bound.

What was wrong with the previous version
----------------------------------------
The earlier script used BFS shortest path on the level-N obstacle set.  Two
problems:
  (1) At level N, the obstacle set is the FULL set C, identical across all
      assignments ℓ.  Random ℓ never entered the path computation, producing
      the same path and the same ε_margin every time.
  (2) BFS shortest path is not the path the VM agent would follow.  ε_margin
      computed along an arbitrary path overestimates the failure rate of an
      agent that actually navigates by Q_mix.

What this version does
----------------------
  (1) Samples ℓ : C → {1, ..., N} and evaluates at an INTERMEDIATE level
      k < N where ℓ produces real variance in the obstacle set.
  (2) Finds the path that the VM would actually traverse: the
      coherence-maximizing widest-path under the FIXED family's Q_mix,
      restricted to the free cells of the level-k test maze.
  (3) The family stays fixed — no Q-table extrapolation needed, in line
      with Theorem 7.2.

Theory tested
-------------
Theorem 7.2:
    Pr_{ℓ_new}[ε_margin > ε] ≤ (d · log(N) + log(1/δ)) / m

where d = d_FI^{ε,δ}.

Usage
-----
    python experiment_b_generalization_bound_v2.py

Imports the family's Q-tables and Q_mix primitives from
navigational_coherence_new.
"""

import os
import sys
import math
import random
import heapq
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set, get_args
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tqdm import tqdm

from GridWorldCanonicalBenchmarkSuite import build_canonical_suite

# Reuse the existing primitives — Q_mix, F* identification, etc.
from MultiAgentMainRecupE import doSeed, load_qtable
from navigational_coherence_new import (
    _identify_consensus_free_space,
    _get_consensus_free_actions,
    _compute_q_mix,
    ACTION_DELTAS,
    State,
)




# =====================================================================
# 1.  Random entry-level assignment
# =====================================================================

def random_entry_level_assignment(obstacle_cells: List[State],
                                 N: int,
                                 rng: random.Random) -> Dict[State, int]:
    """
    Sample a random entry-level assignment ℓ : C → {1,...,N}.

    Each obstacle cell c is independently assigned a density level k ∈ {1,...,N}
    uniformly at random.  Cell c enters the obstacle set at level k:
        O_k = {c ∈ C : ℓ(c) ≤ k}

    Returns dict mapping cell → assigned level.
    """
    return {c: rng.randint(1, N) for c in obstacle_cells}


def obstacles_at_level(assignment: Dict[State, int], k: int) -> Set[State]:
    """
    O_k = {c ∈ C : ℓ(c) ≤ k}   (obstacles present at density level k).
    """
    return {c for c, level in assignment.items() if level <= k}


# =====================================================================
# 2.  VM-aware path finder: coherence-maximizing widest-path
#     restricted to a per-nesting allowed-cell set.
# =====================================================================

def _normalized_margin(q: np.ndarray, action: int,
                       cf_actions: List[int]) -> float:
    """Normalized margin (matches navigational_coherence_new convention)."""
    q_chosen = q[action]
    q_others = [q[a] for a in cf_actions if a != action]
    if not q_others:
        return 1.0
    q_best_other = max(q_others)
    all_q = [q[a] for a in cf_actions]
    q_range = max(all_q) - min(all_q)
    if q_range < 1e-12:
        return 0.0
    return (q_chosen - q_best_other) / q_range


def _consensus_free_actions_in_set(s: State, allowed: Set[State],
                                   W: int, H: int) -> List[int]:
    """Actions whose target is in `allowed`, excluding self-loops."""
    actions = []
    for a, (dx, dy) in enumerate(ACTION_DELTAS):
        tx = max(0, min(W - 1, s[0] + dx))
        ty = max(0, min(H - 1, s[1] + dy))
        target = (tx, ty)
        if target == s:           # self-loop against boundary
            continue
        if target in allowed:
            actions.append(a)
    return actions


def coherence_path_in_allowed_set(
    q_mix_cache: Dict[State, np.ndarray],
    allowed: Set[State],
    start: State,
    goal: State,
    W: int,
    H: int,
) -> Tuple[float, Optional[List[State]], List[float]]:
    """
    Find the path from start to goal that maximizes the minimum normalized
    margin under the FIXED family's Q_mix, restricted to cells in `allowed`.

    `allowed` is typically (family's F*) ∩ (test-maze free cells).  It is
    the per-nesting view of the navigable subspace.

    Returns
    -------
    c_score   : float   max-min margin (the path-level coherence)
    path      : list of states from start to goal, or None if unreachable
    margins   : per-state margins along the returned path
    """
    if start not in allowed or goal not in allowed:
        return float('-inf'), None, []

    INF = float('inf')
    heap = [(-INF, start)]
    best_bottleneck = defaultdict(lambda: float('-inf'))
    best_bottleneck[start] = INF
    came_from: Dict[State, Optional[Tuple[State, int]]] = {start: None}

    while heap:
        neg_bn, s = heapq.heappop(heap)
        bn = -neg_bn
        if bn < best_bottleneck[s]:
            continue
        if s == goal:
            # Reconstruct path and per-step margins
            path = []
            margins: List[float] = []
            cur: Optional[State] = goal
            actions_taken: List[int] = []
            while cur is not None:
                path.append(cur)
                prev = came_from.get(cur)
                if prev is None:
                    break
                actions_taken.append(prev[1])
                cur = prev[0]
            path.reverse()
            actions_taken.reverse()

            # Recompute per-step margins along the chosen path
            for i in range(len(path) - 1):
                ps = path[i]
                a = actions_taken[i]
                cf = _consensus_free_actions_in_set(ps, allowed, W, H)
                q = q_mix_cache.get(ps)
                if q is None:
                    margins.append(float('nan'))
                else:
                    margins.append(_normalized_margin(q, a, cf))

            return (bn if bn != INF else 1.0), path, margins

        cf_actions = _consensus_free_actions_in_set(s, allowed, W, H)
        if not cf_actions:
            continue

        q = q_mix_cache.get(s)
        if q is None:
            continue

        for a in cf_actions:
            dx, dy = ACTION_DELTAS[a]
            s_next = (max(0, min(W - 1, s[0] + dx)),
                      max(0, min(H - 1, s[1] + dy)))
            if s_next == s or s_next not in allowed:
                continue

            margin = _normalized_margin(q, a, cf_actions)
            new_bn = min(bn, margin)
            if new_bn > best_bottleneck[s_next]:
                best_bottleneck[s_next] = new_bn
                came_from[s_next] = (s, a)
                heapq.heappush(heap, (-new_bn, s_next))

    return float('-inf'), None, []


# =====================================================================
# 3.  ε_margin computation
# =====================================================================

def epsilon_margin_from_margins(margins: List[float]) -> float:
    """Fraction of margins on the path that are strictly negative."""
    valid = [m for m in margins if not math.isnan(m)]
    if not valid:
        return 1.0
    return sum(1 for m in valid if m < 0) / len(valid)


# =====================================================================
# 4.  Generalization bound
# =====================================================================

def generalization_bound_basic(d_fi: float, N: int, m: int,
                               delta: float) -> float:
    """
    Basic form of the generalization bound (Theorem 7.2):

        Pr[ε_margin > ε] ≤ (d · log(N) + log(1/δ)) / m
    """
    if m == 0:
        return 1.0
    return (d_fi * math.log(N) + math.log(1.0 / delta)) / m


def generalization_bound_sqrt(d_fi: float, N: int, m: int,
                              delta: float,
                              p_hat: float = 0.0) -> float:
    """
    Hoeffding form (Remark 7.2):

        Pr[ε_margin > ε] ≤ p̂_m + sqrt((d · log(N) + log(1/δ)) / (2m))

    where p̂_m is the empirical failure rate on the m training samples.
    """
    if m == 0:
        return 1.0
    return p_hat + math.sqrt(
        (d_fi * math.log(N) + math.log(1.0 / delta)) / (2.0 * m)
    )


# ===================================================================
# 4.  Main experiment runner
# ===================================================================

@dataclass
class ExperimentBResult:
    """Container for experiment (b) results."""
    grid_size: Tuple[int, int]
    N: int
    eval_level: int
    num_obstacles: int
    m_train: int
    m_test: int
    epsilon_threshold: float
    delta: float

    # Per-sample results (train)
    train_eps_margins: List[float] = field(default_factory=list)
    train_path_lengths: List[int] = field(default_factory=list)
    train_c_scores: List[float] = field(default_factory=list)
    
    # Per-sample results (test)
    test_eps_margins: List[float] = field(default_factory=list)
    test_path_lengths: List[int] = field(default_factory=list)
    test_c_scores: List[float] = field(default_factory=list)
    
    # Aggregate
    train_failure_rate: float = 0.0
    test_failure_rate: float = 0.0
    bound_basic: float = 0.0
    bound_sqrt: float = 0.0
    d_fi_estimate: float = 0.0


def run_experiment_b_v2(
    obstacle_cells: List[State],
    q_tables: List[Dict[State, np.ndarray]],
    W: int, H: int,
    start: State, goal: State,
    eval_level: Optional[int] = None,   # k in {1,...,N}; default = ceil(N/2)
    m_train: int = 50,
    m_test: int = 50,
    epsilon: float = 0.0,
    delta: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> ExperimentBResult:
    """
    Corrected Experiment (b): test the generalization bound using the
    FIXED family's Q_mix and a VM-aware path finder.

    Parameters
    ----------
    obstacle_cells : full obstacle set C from the densest training maze
    q_tables       : N Q-tables (one per training density level)
    W, H           : grid dimensions
    start, goal    : maze endpoints
    eval_level     : level k at which to construct the test maze (1..N).
                     Use an intermediate level so random ℓ produces variance.
                     Defaults to ⌈N/2⌉.
    m_train, m_test : number of training and test nestings to sample
    epsilon, delta : threshold and confidence for the bound
    """

    rng = random.Random(seed)
    N = len(q_tables)
    C_size = len(obstacle_cells)

    if eval_level is None:
        eval_level = (N + 1) // 2
    if not (1 <= eval_level <= N):
        raise ValueError(f"eval_level must be in [1, {N}], got {eval_level}")

    # Pre-compute the FIXED family's F* and Q_mix.
    f_star = _identify_consensus_free_space(q_tables, W, H, start, goal)
    all_cells = {(x, y) for x in range(W) for y in range(H)}    
    q_mix_cache = {s: _compute_q_mix(q_tables, s) for s in all_cells}

    if verbose:
        print(f"{'='*72}")
        print(f"Experiment (b) v2 — Generalization Bound (Theorem 7.2)")
        print(f"{'='*72}")
        print(f"Grid: {W}×{H}  |  N = {N} density levels")
        print(f"|C| = {C_size} obstacle cells  |  start = {start}  goal = {goal}")
        print(f"|F*| (fixed family) = {len(f_star)} cells  "
              f"({len(f_star)/(W*H):.4f} of grid)")
        print(f"Eval level k = {eval_level}  →  expected obstacles per "
              f"nesting ≈ {C_size * eval_level / N:.0f}")
        print(f"m_train = {m_train}  |  m_test = {m_test}")
        print(f"ε = {epsilon}  |  δ = {delta}")
        print()

    result = ExperimentBResult(
        grid_size=(W, H), N=N, eval_level=eval_level,
        num_obstacles=C_size, m_train=m_train, m_test=m_test,
        epsilon_threshold=epsilon, delta=delta,
    )

    def evaluate_nesting(assignment: Dict[State, int]):
        """Build test maze at level k, find VM path, compute ε_margin."""
        obs_at_level = obstacles_at_level(assignment, eval_level)
        # Allowed cells: family F* MINUS the test-maze obstacles
        # allowed = f_star - obs_at_level
        allowed = all_cells - obs_at_level                 # NEW: only test maze obstacles bind        
        # Force include start and goal (they should already be in F* and
        # never in C, but guard against pathological cases)
        allowed.add(start)
        allowed.add(goal)
        # Run VM-aware widest-path
        c_score, path, margins = coherence_path_in_allowed_set(
            q_mix_cache, allowed, start, goal, W, H,
        )
        if path is None:
            return float('-inf'), None, [], 1.0
        eps_m = epsilon_margin_from_margins(margins)
        return c_score, path, margins, eps_m

    # ----- Phase 1: Training set -----
    if verbose:
        print(f"--- Phase 1: Training ({m_train} random nestings) ---")
    train_failures = 0
    for j in range(m_train):
        assignment = random_entry_level_assignment(obstacle_cells, N, rng)
        # print('assignment:', list(assignment.items())[:10])
        c_score, path, margins, eps_m = evaluate_nesting(assignment)
        # print('margins:', margins[:10])
        path_len = (len(path) - 1) if path else 0
        result.train_eps_margins.append(eps_m)
        result.train_path_lengths.append(path_len)
        result.train_c_scores.append(c_score)
        failed = (eps_m > epsilon)
        if failed:
            train_failures += 1
        if verbose and (j + 1) % 10 == 0:
            print(f"  [{j+1:3d}/{m_train}]  ε_margin = {eps_m:.4f}  "
                  f"|P*| = {path_len:>4}  "
                  f"C = {c_score:>7.4f}  "
                  f"{'FAIL' if failed else 'ok'}")

    result.train_failure_rate = train_failures / m_train
    if verbose:
        print(f"\n  Training failure rate: p̂_train = "
              f"{result.train_failure_rate:.4f}  ({train_failures}/{m_train})")

    # ----- Estimate d_FI and compute bounds -----
    if result.train_failure_rate < 1.0 and result.train_failure_rate > 0.0:
        # Use the inverse of the basic bound to estimate d_FI from data:
        # p̂ ≈ (d · log(N) + log(1/δ)) / m  →  d ≈ (p̂ · m - log(1/δ)) / log(N)
        d_fi_est = max(1.0,
            (result.train_failure_rate * m_train - math.log(1.0/delta))
            / max(1.0, math.log(N))
        )
    elif result.train_failure_rate == 0.0:
        # Trivial: bound predicts very low failure rate
        d_fi_est = 1.0
    else:
        # All failed: d_FI ≈ |C|
        d_fi_est = float(C_size)
    result.d_fi_estimate = d_fi_est

    # Compute predicted bounds using both forms
    result.bound_basic = generalization_bound_basic(
        d_fi_est, N, m_train, delta)
    result.bound_sqrt = generalization_bound_sqrt(
        d_fi_est, N, m_train, delta, result.train_failure_rate)

    if verbose:
        print(f"\n  d_FI estimate: {d_fi_est:.1f}")
        print(f"  Predicted test failure rate (basic bound):  "
              f"≤ {min(1.0, result.bound_basic):.4f}")
        print(f"  Predicted test failure rate (sqrt bound):   "
              f"≤ {min(1.0, result.bound_sqrt):.4f}")

    # ----- Phase 2: Test set -----
    if verbose:
        print(f"\n--- Phase 2: Testing ({m_test} held-out nestings) ---")
    test_failures = 0
    for j in range(m_test):
        assignment = random_entry_level_assignment(obstacle_cells, N, rng)
        c_score, path, margins, eps_m = evaluate_nesting(assignment)
        path_len = (len(path) - 1) if path else 0
        result.test_eps_margins.append(eps_m)
        result.test_path_lengths.append(path_len)
        result.test_c_scores.append(c_score)
        failed = (eps_m > epsilon)
        if failed:
            test_failures += 1

        if verbose and (j + 1) % 10 == 0:
            print(f"  [{j+1:3d}/{m_test}]  ε_margin = {eps_m:.4f}  "
                  f"|P*| = {path_len:>4}  "
                  f"C = {c_score:>7.4f}  "
                  f"{'FAIL' if failed else 'ok'}")

    result.test_failure_rate = test_failures / m_test

    # ----- Summary -----
    if verbose:
        print(f"\n{'='*72}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*72}")
        print(f"  Grid: {W}×{H}  |  N = {N}  |  |C| = {C_size}  "
              f"|  k = {eval_level}")
        print(f"  ε threshold: {epsilon}  |  δ: {delta}")
        print()
        print(f"  Training failure rate (m={m_train}):  "
              f"p̂_train = {result.train_failure_rate:.4f}")
        print(f"  Test     failure rate (m={m_test}):   "
              f"p̂_test  = {result.test_failure_rate:.4f}")
        print()
        print(f"  d_FI estimate: {d_fi_est:.1f}")
        print(f"  Bound (basic): Pr[ε_margin > ε] ≤ "
              f"{min(1.0, result.bound_basic):.4f}")
        print(f"  Bound (sqrt):  Pr[ε_margin > ε] ≤ "
              f"{min(1.0, result.bound_sqrt):.4f}")
        print()
        basic_holds = result.test_failure_rate <= min(1.0, result.bound_basic)
        sqrt_holds = result.test_failure_rate <= min(1.0, result.bound_sqrt)
        print(f"  Basic bound holds on test set?  "
              f"{'✓ YES' if basic_holds else '✗ NO'}  "
              f"(p̂_test={result.test_failure_rate:.4f} "
              f"{'≤' if basic_holds else '>'} "
              f"{min(1.0, result.bound_basic):.4f})")
        print(f"  Sqrt  bound holds on test set?  "
              f"{'✓ YES' if sqrt_holds else '✗ NO'}  "
              f"(p̂_test={result.test_failure_rate:.4f} "
              f"{'≤' if sqrt_holds else '>'} "
              f"{min(1.0, result.bound_sqrt):.4f})")
        print()
        train_arr = np.array(result.train_eps_margins)
        test_arr = np.array(result.test_eps_margins)
        train_c = np.array(result.train_c_scores)
        test_c = np.array(result.test_c_scores)
        train_pl = np.array(result.train_path_lengths)
        test_pl = np.array(result.test_path_lengths)
        print(f"  ε_margin (train):  "
              f"mean={train_arr.mean():.4f}  std={train_arr.std():.4f}  "
              f"min={train_arr.min():.4f}  max={train_arr.max():.4f}")
        print(f"  ε_margin (test):   "
              f"mean={test_arr.mean():.4f}  std={test_arr.std():.4f}  "
              f"min={test_arr.min():.4f}  max={test_arr.max():.4f}")
        print(f"  C(V) (train):      "
              f"mean={train_c.mean():.4f}  std={train_c.std():.4f}  "
              f"min={train_c.min():.4f}  max={train_c.max():.4f}")
        print(f"  C(V) (test):       "
              f"mean={test_c.mean():.4f}  std={test_c.std():.4f}  "
              f"min={test_c.min():.4f}  max={test_c.max():.4f}")
        print(f"  |P*| (train):      "
              f"mean={train_pl.mean():.1f}  std={train_pl.std():.1f}  "
              f"min={train_pl.min()}  max={train_pl.max()}")
        print(f"  |P*| (test):       "
              f"mean={test_pl.mean():.1f}  std={test_pl.std():.1f}  "
              f"min={test_pl.min()}  max={test_pl.max()}")
        print(f"{'='*72}")

    return result


# =====================================================================
# 6.  Multi-level sweep: characterize how p̂ depends on eval_level k
# =====================================================================

def     run_experiment_b_v2_multi_level(
    obstacle_cells: List[State],
    q_tables: List[Dict[State, np.ndarray]],
    W: int, H: int,
    start: State, goal: State,
    levels: Optional[List[int]] = None,
    m_train: int = 50, m_test: int = 50,
    epsilon: float = 0.0, delta: float = 0.05,
    seed: int = 42, verbose: bool = True,
) -> Dict[int, ExperimentBResult]:
    """
    Run the corrected experiment across multiple evaluation levels k.
    Useful for tracing how the failure rate depends on the obstacle density
    of the test maze.
    """
    N = len(q_tables)
    if levels is None:
        levels = list(range(1, N + 1))
    results: Dict[int, ExperimentBResult] = {}
    for k in levels:
        if verbose:
            print(f"\n{'#'*72}\n#  Eval level k = {k}\n{'#'*72}")
        results[k] = run_experiment_b_v2(
            obstacle_cells, q_tables, W, H, start, goal,
            eval_level=k, m_train=m_train, m_test=m_test,
            epsilon=epsilon, delta=delta, seed=seed, verbose=verbose,
        )
    return results


# =====================================================================
# 7.  Entry point — example invocation
# =====================================================================

if __name__ == "__main__":
    # The user must construct (q_tables, obstacle_cells, env metadata) using
    # their own loading pipeline (load_qtable + build_canonical_suite).
    # See the header docstring of experiment_c_phase_transition.py for the
    # full pattern; here we sketch the call:

    print("Example invocation skeleton — fill in your loading code:")
    print()
    print("  from your_loader import load_family_qtables, get_obstacle_cells")
    print("  q_tables = load_family_qtables(...)         # list of N dicts")
    print("  W, H = 100, 50")
    print("  start, goal = (0, H-1), (W-1, 0)")
    print("  obstacle_cells = get_obstacle_cells(...)    # full set C")
    print()
    print("  result = run_experiment_b_v2(")
    print("      obstacle_cells, q_tables, W, H, start, goal,")
    print("      eval_level=None,        # default ⌈N/2⌉")
    print("      m_train=50, m_test=50,")
    print("      epsilon=0.0, delta=0.05,")
    print("      seed=42, verbose=True,")
    print("  )")
    print()
    print("To sweep over multiple levels:")
    print()
    print("  results = run_experiment_b_v2_multi_level(")
    print("      obstacle_cells, q_tables, W, H, start, goal,")
    print("      levels=[3, 5, 7, 9],")
    print("      m_train=30, m_test=30,")
    print("  )")

    wall_reward_like_cliff = True
    multiple_starts_perc = 0.2

    episode_mult = 1 # 2 # 
    
    seed = None
    # seed = 2345467781
    seed = doSeed(seed)    

    random_maze_only = True # False # 
    lambda_cliff = 100.0 if random_maze_only else 100.0

    sizes = [(10, 5), (20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    # sizes = [(20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    # sizes = [(40, 20), (60,30), (80, 40), (100, 50)]
    # sizes = [(60,30), (80, 40), (100, 50)]
    # sizes = [(80, 40), (100, 50)]
    sizes = [(100, 50)]
    # sizes = [(10, 5)]
    # sizes = [(20, 10)]
    # sizes = [(40, 20)]
    # sizes = [(60,30)]
    # sizes = [(80, 40)]
    # sizes = [(100, 50)]            
    
    random_maze_str = f'_rmaze{lambda_cliff:.1f}'

    # num_percs = 10
    # perc0 = 0.0
    # percN = 1.0      

    num_percs = 20
    perc0 = 0.0
    percN = 0.8

    suite = build_canonical_suite(sizes, random_maze_only=random_maze_only, lambda_cliff=lambda_cliff, seed=123,
                                      num_percs=num_percs, perc0=perc0, percN=percN)

    envs_eps = [ (env, n_episodes) for (fam, env_name, i), (env, n_episodes) in suite.items() ]

    envs, num_episodes_list = zip(*envs_eps)
    envs_dic = { env_name: env for (fam, env_name, i), env in suite.items() }

    N = num_percs # len(envs)//len(sizes)

    for i, env in enumerate(envs):
        env.update_r_wall(wall_reward_like_cliff)
        env.print_grid(print_rewards=False)

    random_maze_str += f"x{episode_mult}" if episode_mult > 1 else ""

    out_sub_dir = "/traj_self_superv"
    out_sub_dir += random_maze_str + f"-N={N}"
    if not os.path.exists(f"./logs{out_sub_dir}"):
        os.makedirs(f"./logs{out_sub_dir}")
    traj_sub_dir = f"./logs{out_sub_dir}"

    qtable_sub_dir = f"/qtable_self_superv"
    qtable_sub_dir += random_maze_str  + f"-N={N}"   
    if not os.path.exists(f"./logs{qtable_sub_dir}"):
        os.makedirs(f"./logs{qtable_sub_dir}")

    agent_name = 'qlearning'

    default_state_len = len(get_args(State))

    epsilon=0.03
    delta=0.05

    coherence_path = './logs/generalization_bound_report-'+ random_maze_str  + f"-N={N}-p0={perc0}-pN={percN}-eps={epsilon}-delt={delta}.csv"  

    # env = your GridWorldPlus at 100% obstacle density (from RandomMazeFamily)
    # q_tables = list of 20 Q-tables, one per density level

    all_action_space = set()
    qs = []
    for nu, env in enumerate(tqdm(envs)):
        state_len = len(env.reset())
        env.update_r_wall(wall_reward_like_cliff)
        q, action_space = load_qtable(agent_name, env.name, data_dir=f"./logs{qtable_sub_dir}/", 
                                        state_len=state_len, default_state_len=default_state_len)
        qs.append(q)
        all_action_space.update(action_space)

    all_action_space = sorted(list(all_action_space))

    coherence_df = pd.DataFrame()

    print()
    N = len(qs)
   
    env = envs[-1]
    q_tables = qs
    # eval_level = max(1, N//2)

    # --- Step 1: Extract obstacle set C ---
    obstacle_cells = sorted(list(env.walls))
    C_size = len(obstacle_cells)
        
    results = run_experiment_b_v2_multi_level(
        obstacle_cells, q_tables, 
        env.width, env.height, 
        env.start, 
        env.goal,
        #eval_level=eval_level, # None,        # default ⌈N/2⌉
        levels = list(range(1, N)),
        m_train=50, 
        m_test=50,
        epsilon=epsilon, # 0.00, 
        delta=delta, # 0.05,
        seed=42, 
        verbose=True,
    )            

    res_dic_list = []
    for k, result in results.items():
        res_dic = {k:(str(v) if isinstance(v, tuple) else v) for k,v in result.__dict__.items() if not isinstance(v, list)}
        res_df = pd.DataFrame(res_dic, index=[0])
        cols = list(res_df.columns)
        res_df['num_envs'] = N
        res_df['eval_level'] = k
        coherence_df = pd.concat([coherence_df, res_df[['num_envs', 'eval_level']+cols]], ignore_index=True)

    coherence_df.to_csv(coherence_path, index=True, header=True, sep=";", decimal=",")            

    print()
