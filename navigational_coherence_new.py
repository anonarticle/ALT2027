"""
Navigational Coherence Analysis for the Solomonoff Value-Mixture Framework.

Two main functions:

1. compute_q_disagreement(q_tables, W, H, start, goal)
   -> Returns D(s) for all states s in the consensus free space F*.

2. compute_coherence_score(q_tables, W, H, start, goal)
   -> Returns the navigational coherence score C(V) and the optimal path.

These implement the formal definitions from:
  "Solomonoff Value-Mixtures and the Free Inference Boundary", Sections 5.2-5.3.

Q-tables are assumed to be dicts mapping (x, y) -> np.array of shape (4,),
with actions: 0=UP(0,-1), 1=RIGHT(1,0), 2=DOWN(0,1), 3=LEFT(-1,0).

F* identification uses a RELAXED filter: a cell is in F* if it is PRESENT
as a key in ALL Q-tables (regardless of Q-value degeneracy). In a nested
environment family, presence in a Q-table means the cell was visited during
training, which means it is reachable, which means it is free. Degenerate
Q-values (e.g., all actions equal) signal incomplete training, not obstacle
status. The goal state (terminal, Q=[0,0,0,0]) is the canonical example.
"""

from matplotlib import pyplot as plt
import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import pandas as pd
from tqdm import tqdm 
import os 

# from navigational_coherence import coherence_report
from GridWorldCanonicalBenchmarkSuite import build_canonical_suite
from MultiAgentMainRecup import load_qtable

State = Tuple[int, int]
ACTION_DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT

just_plot_charts = False # True # 

# ---------------------------------------------------------------------------
# Helper: identify consensus free space F* from Q-tables
# ---------------------------------------------------------------------------

def _identify_consensus_free_space(
    q_tables: List[Dict[State, np.ndarray]],
    W: int,
    H: int,
    start: Optional[State] = None,
    goal: Optional[State] = None,
) -> Set[State]:
    """
    Identify the consensus free space F*: the set of cells (x, y) that are
    PRESENT in ALL Q-tables.

    A cell is considered "present" in a Q-table if:
      - It exists as a key in the Q-table,
      - It has a valid Q-value array of length 4,
      - Its coordinates are within grid bounds.

    No filter on Q-value magnitude or variance is applied: presence alone
    is sufficient evidence that the cell is free (was visited during training).

    Start and goal states are force-included if provided.

    F* = intersection of present cells across all Q-tables.
    """
    N = len(q_tables)
    if N == 0:
        return set()

    # For each Q-table, find the set of present states
    present_per_env = []
    for Q in q_tables:
        present = set()
        for s, qvals in Q.items():
            if qvals is None or len(qvals) != 4:
                continue
            x, y = s[0], s[1]
            if not (0 <= x < W and 0 <= y < H):
                continue
            present.add((x, y))
        present_per_env.append(present)

    # F* = intersection of all present sets
    f_star = present_per_env[0]
    for p in present_per_env[1:]:
        f_star = f_star & p

    # Force-include start and goal
    if start is not None:
        f_star.add(start)
    if goal is not None:
        f_star.add(goal)

    return f_star


def _get_consensus_free_actions(s: State, f_star: Set[State], W: int, H: int) -> List[int]:
    """
    Return the list of actions from state s whose target cell is in F*
    and within grid bounds, excluding self-loops.
    """
    x, y = s
    actions = []
    for a, (dx, dy) in enumerate(ACTION_DELTAS):
        tx = max(0, min(W - 1, x + dx))
        ty = max(0, min(H - 1, y + dy))
        if (tx, ty) in f_star and (tx, ty) != (x, y):
            # Exclude self-loops (action against boundary that doesn't move)
            actions.append(a)
    return actions


def _compute_q_mix(
    q_tables: List[Dict[State, np.ndarray]],
    s: State,
) -> np.ndarray:
    """
    Compute Q_mix(s, a) = (1/N_present) * sum_i Q_i(s, a) for all 4 actions.
    Only averages over environments that have the state.
    Returns np.zeros(4) if state not found in any Q-table.
    """
    q_mix = np.zeros(4, dtype=np.float64)
    count = 0
    for Q in q_tables:
        qvals = Q.get(s)
        if qvals is not None and len(qvals) == 4:
            q_mix += np.asarray(qvals, dtype=np.float64)
            count += 1
    if count > 0:
        q_mix /= count
    return q_mix


# ---------------------------------------------------------------------------
# Function 1: Q-Value Disagreement D(s)
# ---------------------------------------------------------------------------

def compute_q_disagreement(
    q_tables: List[Dict[State, np.ndarray]],
    W: int,
    H: int,
    start: Optional[State] = None,
    goal: Optional[State] = None,
) -> Dict[State, float]:
    """
    Compute the Q-value disagreement D(s) for all states s in F*.

    Definition (from Section 5.3):

        D(s) = max_{a, a' in A_F*(s)} Var_{nu ~ Uniform(V)} [Q_nu(s,a) - Q_nu(s,a')]

    where A_F*(s) is the set of actions from s whose target is in F*.

    D(s) measures how much the environments disagree about the relative
    ranking of consensus-free actions at state s. When D(s) = 0, all
    environments agree on action preferences. When D(s) is large,
    environments strongly disagree about which direction to go within F*.

    Parameters
    ----------
    q_tables : list of dicts
        Each dict maps (x, y) -> np.array of shape (4,).
    W, H : int
        Grid width and height.
    start, goal : (x, y) or None
        Start and goal states (force-included in F*).

    Returns
    -------
    disagreement : dict mapping (x, y) -> float
        D(s) for each state s in F*. States not in F* are excluded.
    """
    N = len(q_tables)
    f_star = _identify_consensus_free_space(q_tables, W, H, start, goal)
    disagreement = {}

    for s in f_star:
        # Get consensus-free actions at this state
        cf_actions = _get_consensus_free_actions(s, f_star, W, H)

        if len(cf_actions) < 2:
            # With 0 or 1 consensus-free actions, no disagreement is possible
            disagreement[s] = 0.0
            continue

        # Collect Q-values from each environment for this state
        # q_all[nu][a] = Q_nu(s, a) for consensus-free actions
        q_all = []
        for Q in q_tables:
            qvals = Q.get(s)
            if qvals is not None and len(qvals) == 4:
                q_all.append(np.asarray(qvals, dtype=np.float64))
            else:
                q_all.append(np.zeros(4, dtype=np.float64))

        # Compute D(s) = max over all pairs (a, a') in A_F*(s)
        # of Var_{nu}[Q_nu(s,a) - Q_nu(s,a')]
        max_var = 0.0
        for i, a in enumerate(cf_actions):
            for a_prime in cf_actions[i + 1:]:
                # Compute the difference Q_nu(s,a) - Q_nu(s,a') for each env
                diffs = np.array([q_all[nu][a] - q_all[nu][a_prime]
                                  for nu in range(N)])
                var = float(np.var(diffs))
                if var > max_var:
                    max_var = var

        disagreement[s] = max_var

    return disagreement


# ---------------------------------------------------------------------------
# Function 2: Navigational Coherence Score C(V)
# ---------------------------------------------------------------------------

def compute_coherence_score(
    q_tables: List[Dict[State, np.ndarray]],
    W: int,
    H: int,
    start: State,
    goal: State,
) -> Tuple[float, Optional[List[State]]]:
    """
    Compute the navigational coherence score C(V) and the optimal path.

    Definition (from Section 5.3):

        C(V) = max_{P: s0 -> g in F*} min_{s in P} margin(s, a_P(s))

    where:
        margin(s, a) = [Q_mix(s,a) - max_{a'!=a} Q_mix(s,a')] 
                        / [max_a Q_mix(s,.) - min_a Q_mix(s,.)]

    This is the normalized margin of the path-continuing action over the
    best alternative. C(V) > 0 means there exists a path where Q_mix's
    greedy policy has positive margin at every state (Free Inference holds).
    C(V) <= 0 means every path has at least one state where Q_mix prefers
    a different direction (Free Inference fails).

    The computation uses a max-min (widest-path / bottleneck) algorithm:
    a modified Dijkstra that maximizes the minimum edge weight along the
    path from start to goal.

    Parameters
    ----------
    q_tables : list of dicts
        Each dict maps (x, y) -> np.array of shape (4,).
    W, H : int
        Grid width and height.
    start : (x, y)
        Start state.
    goal : (x, y)
        Goal state.

    Returns
    -------
    coherence_score : float
        C(V). Positive means Free Inference holds, non-positive means it fails.
        Returns -inf if no path exists in F* from start to goal.
    best_path : list of (x, y) or None
        The path achieving C(V), or None if no path exists.
    """
    f_star = _identify_consensus_free_space(q_tables, W, H, start, goal)

    if start not in f_star or goal not in f_star:
        return float('-inf'), None

    # Precompute Q_mix for all states in F*
    q_mix_cache = {}
    for s in f_star:
        q_mix_cache[s] = _compute_q_mix(q_tables, s)

    # Build the graph: for each state s in F*, for each consensus-free
    # action a, compute the normalized margin and the target state.
    #
    # Edge: (s, a) -> s', with weight = normalized_margin(s, a)
    # normalized_margin(s, a) = (Q_mix(s,a) - max_{a'!=a} Q_mix(s,a'))
    #                           / (max Q_mix(s,.) - min Q_mix(s,.))
    #
    # Note: we restrict to actions whose target is in F*.

    def compute_normalized_margin(s: State, action: int, cf_actions: List[int]) -> float:
        """Compute the normalized margin for choosing 'action' at state s,
        considering only consensus-free actions."""
        q = q_mix_cache[s]

        # Q_mix value for the chosen action
        q_chosen = q[action]

        # Best Q_mix value among OTHER consensus-free actions
        q_others = [q[a] for a in cf_actions if a != action]
        if len(q_others) == 0:
            # Only one consensus-free action: margin is maximal (no alternative)
            return 1.0
        q_best_other = max(q_others)

        # Normalization: range of Q_mix over consensus-free actions
        all_q = [q[a] for a in cf_actions]
        q_range = max(all_q) - min(all_q)

        if q_range < 1e-12:
            # All consensus-free actions have identical Q_mix: zero margin,
            # but not negative (the agent doesn't prefer a wrong direction,
            # it just can't distinguish directions)
            return 0.0

        return (q_chosen - q_best_other) / q_range

    def target_of(s: State, a: int) -> State:
        dx, dy = ACTION_DELTAS[a]
        return (max(0, min(W - 1, s[0] + dx)),
                max(0, min(H - 1, s[1] + dy)))

    # --- Max-min (widest-path) Dijkstra ---
    #
    # We want to find the path from start to goal in F* that maximizes
    # the minimum normalized margin along the path.
    #
    # State in the algorithm: (bottleneck, current_cell)
    # where bottleneck = minimum margin encountered so far on the path.
    #
    # We use a max-heap (negate values for Python's min-heap).
    # At each step, expand the node with the largest bottleneck.
    # When we reach the goal, that bottleneck is C(V).

    # Priority queue: (-bottleneck, state)
    # Start with bottleneck = +inf (no constraint yet)
    INF = float('inf')
    # Start node: bottleneck = +inf (no constraint yet)
    heap = [(-INF, start)]

    # Best bottleneck found for reaching each state
    best_bottleneck = defaultdict(lambda: float('-inf'))
    best_bottleneck[start] = INF

    # For path reconstruction
    came_from = {start: None}  # state -> (prev_state, action)

    while heap:
        neg_bn, s = heapq.heappop(heap)
        bn = -neg_bn  # current bottleneck (minimum margin so far)

        # If we've already found a better path to s, skip
        if bn < best_bottleneck[s]:
            continue

        # If we've reached the goal, return the result
        if s == goal:
            # Reconstruct path
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                prev = came_from.get(cur)
                cur = prev[0] if prev is not None else None
            path.reverse()
            return bn if bn != INF else 1.0, path

        # Expand neighbors
        cf_actions = _get_consensus_free_actions(s, f_star, W, H)
        if len(cf_actions) == 0:
            continue

        for a in cf_actions:
            s_next = target_of(s, a)

            # Skip self-loops (action against boundary)
            if s_next == s:
                continue
            if s_next not in f_star:
                continue

            # Compute margin for taking action a at state s
            margin = compute_normalized_margin(s, a, cf_actions)

            # New bottleneck = min(current bottleneck, this edge's margin)
            new_bn = min(bn, margin)

            # Only expand if this is better than what we've seen for s_next
            if new_bn > best_bottleneck[s_next]:
                best_bottleneck[s_next] = new_bn
                came_from[s_next] = (s, a)
                heapq.heappush(heap, (-new_bn, s_next))

    # Goal not reachable through F*
    return float('-inf'), None


# ---------------------------------------------------------------------------
# Convenience: summary report
# ---------------------------------------------------------------------------

def coherence_report(
    q_tables: List[Dict[State, np.ndarray]],
    densities: List[str],
    W: int,
    H: int,
    start: State,
    goal: State,
    env_names: Optional[List[str]] = None,
) -> dict:
    """
    Compute and print a full coherence analysis report.

    Returns a dict with keys:
        'f_star_size': number of cells in F*
        'disagreement': dict of D(s) for all s in F*
        'coherence_score': C(V)
        'best_path': the path achieving C(V)
        'mean_disagreement': average D(s) over F*
        'max_disagreement': maximum D(s) over F*
        'high_disagreement_states': states where D(s) > mean + 2*std
    """
    print("=" * 70)
    print("NAVIGATIONAL COHERENCE ANALYSIS")
    print("=" * 70)

    # Identify F*
    f_star = _identify_consensus_free_space(q_tables, W, H, start, goal)
    print(f"\nGrid: {W} x {H} = {W * H} cells")
    print(f"Number of environments: {len(q_tables)}")
    if env_names:
        print(f"Environments: {', '.join(env_names)}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Consensus free space |F*|: {len(f_star)} cells")
    print(f"F* density: {len(f_star) / (W * H):.1%}")
    print(f"Max density: {densities[-1]:.1%}")

    # Compute Q-value disagreement
    print(f"\n--- Q-Value Disagreement D(s) ---")
    disagreement = compute_q_disagreement(q_tables, W, H, start, goal)

    d_values = np.array(list(disagreement.values()))
    if len(d_values) > 0:
        d_mean = float(d_values.mean())
        d_std = float(d_values.std())
        d_max = float(d_values.max())
        d_median = float(np.median(d_values))
    else:
        d_mean = d_std = d_max = d_median = 0.0

    print(f"  Mean D(s):   {d_mean:.4f}")
    print(f"  Median D(s): {d_median:.4f}")
    print(f"  Std D(s):    {d_std:.4f}")
    print(f"  Max D(s):    {d_max:.4f}")

    # High-disagreement states
    threshold = d_mean + 2 * d_std if d_std > 0 else d_mean + 1e-6
    high_d_states = {s: d for s, d in disagreement.items() if d > threshold}
    print(f"  States with D(s) > mean + 2*std ({threshold:.4f}): {len(high_d_states)}")

    if len(high_d_states) > 0:
        sorted_high = sorted(high_d_states.items(), key=lambda x: -x[1])
        print(f"  Top-10 highest disagreement states:")
        for s, d in sorted_high[:10]:
            print(f"    {s}: D = {d:.4f}")

    # Compute coherence score
    print(f"\n--- Navigational Coherence Score C(V) ---")
    c_score, best_path = compute_coherence_score(q_tables, W, H, start, goal)

    print(f"  C(V) = {c_score:.6f}")
    if c_score > 0:
        print(f"  Interpretation: FREE INFERENCE HOLDS (positive margin)")
        c_score_interpret = 'FREE INFERENCE HOLDS'
    elif c_score == float('-inf'):
        print(f"  Interpretation: NO PATH through F* from {start} to {goal}")
        c_score_interpret = 'NO PATH through F*'
    elif c_score > -1e-9:
        print(f"  Interpretation: MARGINAL (zero margin at bottleneck)")
        c_score_interpret = 'MARGINAL'
    else:
        print(f"  Interpretation: FREE INFERENCE FAILS (negative margin)")
        c_score_interpret = 'FREE INFERENCE FAILS'

    if best_path is not None:
        print(f"  Best path length: {len(best_path)} steps")

        # Find the bottleneck state(s) along the path
        q_mix_cache = {s: _compute_q_mix(q_tables, s) for s in f_star}
        min_margin = float('inf')
        bottleneck_state = None
        for i, s in enumerate(best_path[:-1]):  # exclude goal
            s_next = best_path[i + 1]
            # Find which action leads from s to s_next
            for a, (dx, dy) in enumerate(ACTION_DELTAS):
                tx = max(0, min(W - 1, s[0] + dx))
                ty = max(0, min(H - 1, s[1] + dy))
                if (tx, ty) == s_next:
                    cf_actions = _get_consensus_free_actions(s, f_star, W, H)
                    q = q_mix_cache[s]
                    q_chosen = q[a]
                    q_others = [q[aa] for aa in cf_actions if aa != a]
                    if len(q_others) > 0:
                        q_best_other = max(q_others)
                        all_q = [q[aa] for aa in cf_actions]
                        q_range = max(all_q) - min(all_q)
                        if q_range > 1e-12:
                            m = (q_chosen - q_best_other) / q_range
                        else:
                            m = 0.0
                    else:
                        m = 1.0
                    if m < min_margin:
                        min_margin = m
                        bottleneck_state = s
                    break

        if bottleneck_state is not None:
            print(f"  Bottleneck state: {bottleneck_state} (margin = {min_margin:.6f})")
            print(f"  Bottleneck Q_mix: {q_mix_cache[bottleneck_state]}")
    else:
        print(f"  No path found from {start} to {goal} through F*")

    print("=" * 70)

    return {
        'env_size': f'{W}x{H}',
        'number_of_envs': len(q_tables),
        'f_star_size': len(f_star),
        'f_star_density': len(f_star) / (W * H),
        'max_density': densities[-1],
        'disagreement_mean':  d_mean,
        'disagreement_median':d_median,
        'disagreement_std':   d_std,
        'disagreement_max':   d_max,
        'coherence_score': c_score,
        'c_score_interpret': c_score_interpret,
        'best_path_len': 0 if best_path is None else len(best_path),
        'number_of_high_disagreement_states': len(high_d_states),
    }

if __name__ == "__main__":

    wall_reward_like_cliff = True
    random_maze_only = True # False # 
    lambda_cliff = 100.0 if random_maze_only else 100.0
    default_state_len = 2
    episode_mult = 1 # 2 # 


    num_percs = 10 # 20
    perc0 = 0.0
    percN = 1.0      

    # num_percs = 40
    # perc0 = 0.5
    # percN = 0.75     
                    
    agent_name = 'qlearning'

    nested_grids = False #  True # 

    sizes = [(10, 5), (20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    # sizes = [(20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    # sizes = [(40, 20), (60,30), (80, 40), (100, 50)]
    # sizes = [(60,30), (80, 40), (100, 50)]
    # sizes = [(80, 40), (100, 50)]
    # sizes = [(100, 50)]
    # sizes = [(10, 5)]
    # sizes = [(20, 10)]
    sizes = [(40, 20)]
    # sizes = [(60,30)]
    # sizes = [(80, 40)]
    # sizes = [(100, 50)]            

    random_maze_str = f'_rmaze{lambda_cliff:.1f}'

    suite = build_canonical_suite(sizes, random_maze_only=random_maze_only, lambda_cliff=lambda_cliff, seed=123,
                                    num_percs=num_percs, nested_grids=nested_grids, perc0=perc0, percN=percN)

    envs_eps = [ (env, n_episodes) for (fam, env_name, i), (env, n_episodes) in suite.items() ]

    envs, num_episodes_list = zip(*envs_eps)
    envs_dic = { env_name: env for (fam, env_name, i), env in suite.items() }

    N = len(envs)//len(sizes) 

    for i, env in enumerate(envs):
        env.update_r_wall(wall_reward_like_cliff)
        env.print_grid(print_rewards=False)

    random_maze_str += f"x{episode_mult}" if episode_mult > 1 else ""

    out_sub_dir = "/traj_self_superv"
    out_sub_dir += random_maze_str + f"-N={N}" + ("-ng" if nested_grids else "-nn")
    if not os.path.exists(f"./logs{out_sub_dir}"):
        os.makedirs(f"./logs{out_sub_dir}")
    traj_sub_dir = f"./logs{out_sub_dir}"

    qtable_sub_dir = f"/qtable_self_superv"
    qtable_sub_dir += random_maze_str  + f"-N={N}" + ("-ng" if nested_grids else "-nn")   
    if not os.path.exists(f"./logs{qtable_sub_dir}"):
        os.makedirs(f"./logs{qtable_sub_dir}")

    sizes0 = [(10, 5), (20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    # sizes0 = [(80, 40), (100, 50)]
    # sizes0 = [(100, 50)]
    sizes0 = [(40, 20)]

    coherence_df = pd.DataFrame()

    if (perc0, percN) == (0.0, 1.0):
        sub_sampling_env_nums = [2, 5, 10, 20]
    else:
        sub_sampling_env_nums = [num_percs]
    max_num_envs = sub_sampling_env_nums[-1]
    num_sub_samplings = len(sub_sampling_env_nums)

    coherence_path = './logs/coeherence_report-'+ random_maze_str  + f"{"-ng" if nested_grids else "-nn"}-S={num_sub_samplings}.csv"  

    if not just_plot_charts:
        for num_envs in sub_sampling_env_nums:

            env_idxs = list(range(max_num_envs//num_envs-1,max_num_envs, max_num_envs//num_envs))
            
            for size in sizes0[:]:
                all_action_space = set()   

                sizes = [size]
                suite = build_canonical_suite(sizes, random_maze_only=random_maze_only, lambda_cliff=lambda_cliff, seed=123,
                                                num_percs=num_percs, nested_grids=nested_grids, perc0=perc0, percN=percN)

                envs_eps = [ (env, n_episodes) for (fam, env_name, i), (env, n_episodes) in suite.items() ]

                envs, num_episodes_list = zip(*envs_eps)
                envs_dic = { env_name: env for (fam, env_name, i), env in suite.items() }

                qs = []
                densities =  []
                for nu, env in enumerate(tqdm(envs[:])):

                    if nu in env_idxs:
                        state_len = len(env.reset())
                        env.update_r_wall(wall_reward_like_cliff)
                        q, action_space = load_qtable(agent_name, env.name, data_dir=f"./logs{qtable_sub_dir}/", 
                                                        state_len=state_len, default_state_len=default_state_len, suffix="")
                        qs.append(q)
                        all_action_space.update(action_space)
                        density = env.name.split('-')[2] if len(env.name.split('-')) > 3 else '1.0'
                        densities.append(float(density))

                all_action_space = sorted(list(all_action_space))

                # q_tables is your list of Q-table dicts, one per environment
                # env is any environment template (for W, H, start, goal)
                print()
                for i in range(len(qs)):
                    result = coherence_report(
                        q_tables=qs[:i+1],
                        densities=densities[:i+1],
                        W=env.width,
                        H=env.height,
                        start=env.start,
                        goal=env.goal,
                    )
                    print()
                    res_df = pd.DataFrame(result, index=[0])
                    cols = list(res_df.columns)
                    res_df['num_envs'] = num_envs
                    coherence_df = pd.concat([coherence_df, res_df[['num_envs']+cols]], ignore_index=True)

        coherence_df.to_csv(coherence_path, index=True, header=True, sep=";", decimal=",")
    else:
        coherence_df = pd.read_csv(coherence_path, index_col=0, header=[0], sep=";", decimal=",")

    plot_markers = [
    # '.', # Point marker
    # ',', # Pixel marker
    'o', # Circle marker
    's', # Square marker
    '^', # Triangle up marker
    'p', # Pentagon marker
    'D', # Diamond marker
    'v', # Triangle down marker
    '*', # Star marker
    '<', # Triangle left marker
    '>', # Triangle right marker
    'x', # X marker
    '+', # Plus marker
    'P', # Plus (filled) marker
    'h', # Hexagon 1 marker
    'H', # Hexagon 2 marker
    'X', # X (filled) marker
    'd', # Thin diamond marker    
    ]

    env_sizes = coherence_df['env_size'].unique()

    cols=2 - (num_sub_samplings==1)*1
    rows=(num_sub_samplings+cols-1)//cols
    hspace=0.25
    wspace=0.15
    # fig, axs = plt.subplots(rows, cols, figsize=(cols*6*(1+wspace*2), rows*5*(1+hspace)), constrained_layout=True)  
    fig, axs = plt.subplots(rows, cols, figsize=(7, 6), constrained_layout=True)  

    # fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)

    fig.suptitle('Coherence score', fontsize=12)

    chrc_df = coherence_df.set_index(['num_envs', 'env_size'])

    for ai, ne in enumerate(sub_sampling_env_nums):

        if rows == 1 and cols == 1:
            ax = axs
        elif rows == 1 or cols == 1:
            ax = axs[ai]        
        else:
            ax = axs[ai//cols, ai%cols]
        m = 0
        for env_size in env_sizes:
            c_df = chrc_df.loc[(ne, env_size)]
            ax.plot(c_df['max_density'], c_df['coherence_score'], marker=plot_markers[m], label=f'rmaze-{env_size}')

            m += 1

            ax.set_title(f'N = {len(c_df)}', fontsize=12)
            ax.legend(loc='lower left', fontsize=6)
            ax.set_xticks(c_df['max_density'])
            ax.set_xticklabels(c_df['max_density'], )
            ax.set_xlim(perc0-(perc0 != 0)*0.05, percN+0.05)
            ax.set_ylim(-1.02,0.02)
            # ax.set_yticks(np.array(range(21))/20-1)
            # ax.xaxis.tick_top()
            ax.tick_params(axis='x', labelrotation=45, labelsize=6)             
            ax.tick_params(axis='y', labelsize=6) 
            ax.grid(True)

    plt.savefig(coherence_path[:-4]+'.png')
    plt.show()
    print()

    # result['coherence_score'] > 0  means Free Inference holds
    # result['disagreement']         is the D(s) dict for visualization
    # result['best_path']            is the optimal coherence path