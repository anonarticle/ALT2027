"""
fixed_rho_sweep_helpers.py

Companion to fixed_rho_sweep.py.  Provides:
  - train_qtables_for_envs : per-env Q-learning + cache
  - run_vm_pms_rollouts    : VM and PMS rollouts with bps tracking
"""

import os
import math
from collections import defaultdict
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from Q_learning_Agent_plus_plus import q_learning, get_trajectory
from MetaAgent2NIGSelfSup import (
    MetaAgent2NIGSelfSup as MetaAgent,
    ACTION_DELTAS,
)
from MultiAgentMainRecupE import load_qtable


# =================================================================
# 1.  Q-table training and on-disk caching
# =================================================================

def train_qtables_for_envs(
    envs,
    qtable_dir,
    agent_name='qlearning',
    num_episodes=500,
    alpha=0.1,
    gamma=1.0,
    epsilon_start=0.1,
    epsilon_end=0.05,
    state_len=2,
    default_state_len=2,
    use_iterative=True,
    force_retrain=False,
    show_progress=True,
    just_train_qtables=False,
):
    """
    Train and persist one Q-table per env in `envs`, in the on-disk
    format that load_qtable() expects.  Skips envs whose CSV already
    exists unless force_retrain=True.

    Returns
    -------
    list of (Q, action_space) tuples in the same order as `envs`.
    Q is a defaultdict[state -> np.ndarray of length 4].

    Notes on use_iterative
    ----------------------
    The production loop in MultiAgentMainRecupE.py expands start
    states by Manhattan distance and re-trains until each one's
    greedy trajectory reaches the goal.  This matters at high
    obstacle density (>= 0.5), where vanilla random-start Q-learning
    can leave large regions of state space underfit.

    For the fixed-rho sweep at rho in {0.10, 0.20, 0.30}, the simple
    path (use_iterative=False) is usually adequate and ~5-10x faster.
    Use the iterative path if you see Q-tables with all-zero rows
    near the goal, or if you scale the sweep to higher rho.
    """
    os.makedirs(qtable_dir, exist_ok=True)
    results = []

    iterator = tqdm(envs, desc='train_qtables') if show_progress else envs
    for env in iterator:
        Q_path = _qtable_path(
            qtable_dir, agent_name, env.name, state_len, default_state_len,
        )

        if os.path.exists(Q_path) and not force_retrain:
            if not just_train_qtables:
                results.append(_reload(qtable_dir, agent_name, env.name,
                                    state_len, default_state_len))
            continue

        print(f'env: {env.name}')
        Q_df = pd.DataFrame(columns=["dummy"], index=[0, 1])
        Q_df.to_csv(Q_path,index=False)
    
        if use_iterative:
            q = _train_iterative(
                env,
                num_episodes=num_episodes,
                alpha=alpha, gamma=gamma,
                epsilon_start=epsilon_start, epsilon_end=epsilon_end,
            )
        else:
            q, _, _ = q_learning(
                env,
                num_episodes=num_episodes,
                alpha=alpha, gamma=gamma,
                epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                prev_Q=None,
                start_state=None,                           # random starts
                max_steps=10 * env.width * env.height,
                use_tqdm=False,
            )

        action_space = list(range(len(ACTION_DELTAS)))
        _save_qtable(q, env, Q_path, action_space)

        if not just_train_qtables:
            # Re-load so callers get the exact same dict layout that
            # load_qtable() would have produced (defaultdict, sorted cols).
            results.append(_reload(qtable_dir, agent_name, env.name,
                                state_len, default_state_len))

    return results


def _qtable_path(qtable_dir, agent_name, env_name,
                 state_len, default_state_len, suffix=''):
    suf = f'-{state_len}' if state_len != default_state_len else ''
    return os.path.join(qtable_dir,
                        f'{agent_name}_{env_name}{suf}_qtable{suffix}.csv')


def _reload(qtable_dir, agent_name, env_name, state_len, default_state_len):
    return load_qtable(
        agent_name, env_name,
        data_dir=qtable_dir,
        state_len=state_len,
        default_state_len=default_state_len,
        suffix='',
    )


def _save_qtable(q, env, Q_path, action_space):
    """Mirror the on-disk format used in MultiAgentMainRecupE.py:
       columns = state_columns() + action_cols, indexed sequentially."""
    Q_df = pd.DataFrame(q).T
    action_cols = list(Q_df.columns)
    Q_df = Q_df.reset_index()
    Q_df.columns = list(env.state_columns()) + action_cols
    Q_df.to_csv(Q_path, index=True, header=True)


def _train_iterative(env, num_episodes, alpha, gamma,
                     epsilon_start, epsilon_end, max_retries=3):
    """
    Compressed version of the production iterative loop:
    expand start states by Manhattan distance to goal, retrain until
    the greedy trajectory from each start reaches the goal.
    """
    max_distance = env.width + env.height
    sorted_states = sorted(
        env.all_states(),
        key=lambda x: sum((a - b) ** 2 for a, b in zip(x, env.goal)),
    )[1:]                                                   # skip goal

    q = None
    last_good_q = None
    n_actions = len(ACTION_DELTAS)

    schedule = sorted_states + [env.start] * 2

    for s in tqdm(schedule):
        if s == env.goal or env.forbidden(s):
            continue

        state_dist = sum(abs(a - b) for a, b in zip(s, env.goal))
        ep_perc = max(0.1, state_dist / max_distance)

        for _ in range(max_retries):
            q, _, _ = q_learning(
                env,
                num_episodes=int(num_episodes * ep_perc),
                alpha=alpha, gamma=gamma,
                epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                prev_Q=q,
                start_state=s,
                max_steps=-1,
                use_tqdm=False,
            )
            best_traj, _ = get_trajectory(
                env, q, greedy=True,
                max_steps=5 * env.width * env.height,
                start_state=s,
            )
            if best_traj and best_traj[-1] == env.goal:
                last_good_q = defaultdict(lambda: np.zeros(n_actions),
                                          dict(q))
                break
            else:
                # Roll back to the last known-good Q-table and retry
                q = (defaultdict(lambda: np.zeros(n_actions),
                                 dict(last_good_q))
                     if last_good_q is not None else None)

    return last_good_q if last_good_q is not None else q


# =================================================================
# 2.  VM + PMS rollouts on the family
# =================================================================

def run_vm_pms_rollouts(
    envs,
    qs,
    action_space,
    n_episodes_per_env=1,
    max_steps_factor=5,
    greedy=True,
    normalize=True,
    epsilon=0.0,
    seed=None,
    record_per_step=False,
):
    """
    For each env_i in `envs` treated as the true environment nu*,
    run one VM episode and one PMS episode of the meta-agent built
    from the family `qs` of Q-tables.

    Returns
    -------
    dict with:
      'per_run'  : long-format DataFrame, one row per
                   (mode, true_env_idx, episode).
      'per_step' : long-format DataFrame, one row per step.  Only
                   present when record_per_step=True.
      'summary'  : DataFrame, one row per mode, with the headline
                   numbers that go into the sweep CSV.

    Per-run columns
    ---------------
      mode            'vm' | 'pms'
      true_env_idx    int    (index into envs/qs)
      true_env_name   str
      episode         int
      steps           int    (# env steps until done or max_steps)
      reached_goal    bool
      total_return    float
      collisions      int    (# steps where r == env.r_wall)
      bps_mean        float
      bps_sum         float
      kl_to_uniform   float  (final belief KL to uniform prior)
      hit_cells       int    (|MetaAgent.obstacle_hits|)
    """
    per_run_rows = []
    per_step_rows = []

    W, H = envs[0].width, envs[0].height
    max_steps = max_steps_factor * W * H

    mode_pairs = [('vm',  'value-mixture'),
                  ('pms', 'thompson')]      # 'thompson' is PMS internally

    for true_idx, true_env in enumerate(envs):
        for mode_name, acting_mode in mode_pairs:
            for ep in range(n_episodes_per_env):
                # Fresh meta-agent per rollout: ensures belief reset and
                # per-cell NIG models start from prior.  reset_belief()
                # also handles this, but a fresh constructor is more
                # defensive against any future stateful additions.
                meta = MetaAgent(
                    env_template=envs[0],
                    q_tables=qs,
                    action_space=action_space,
                    epsilon=epsilon,
                    use_belief_weighted_Q=(mode_name == 'vm'),
                    seed=seed,
                )

                summary, step_recs = _rollout_one(
                    meta=meta,
                    true_env=true_env,
                    acting_mode=acting_mode,
                    greedy=greedy,
                    normalize=normalize,
                    max_steps=max_steps,
                    record_per_step=record_per_step,
                )
                summary.update({
                    'mode':          mode_name,
                    'true_env_idx':  true_idx,
                    'true_env_name': getattr(true_env, 'name', str(true_idx)),
                    'episode':       ep,
                })
                per_run_rows.append(summary)

                if record_per_step:
                    for r in step_recs:
                        r.update({
                            'mode':          mode_name,
                            'true_env_idx':  true_idx,
                            'true_env_name': getattr(true_env, 'name',
                                                     str(true_idx)),
                            'episode':       ep,
                        })
                        per_step_rows.append(r)

    per_run = pd.DataFrame(per_run_rows)
    summary = (per_run
               .groupby('mode')
               .agg(return_mean      = ('total_return', 'mean'),
                    return_std       = ('total_return', 'std'),
                    collisions_mean  = ('collisions',   'mean'),
                    collisions_std   = ('collisions',   'std'),
                    bps_mean         = ('bps_mean',     'mean'),
                    bps_std          = ('bps_mean',     'std'),
                    success_rate     = ('reached_goal', 'mean'),
                    kl_final_mean    = ('kl_to_uniform','mean'),
                    n_runs           = ('episode',      'count'))
               .reset_index())

    out = {'per_run': per_run, 'summary': summary}
    if record_per_step:
        out['per_step'] = pd.DataFrame(per_step_rows)
    return out


def _rollout_one(meta, true_env, acting_mode, greedy, normalize,
                 max_steps, record_per_step):
    """
    Single episode of `meta` on `true_env` with the given acting mode.

    Inlines MetaAgent.run_episode() so we can compute per-step bps
    via log_prob_action() alongside the action choice.

    Subtlety: act() and log_prob_action() both bump
    meta.action_count[s][action] as a tie-breaker.  Calling them in
    sequence double-bumps.  We snapshot/restore action_count around
    the log_prob_action call to keep behaviour identical to a normal
    run_episode while still measuring bps at the correct (s, a).
    """
    meta.reset_belief()
    meta.use_belief_weighted_Q = (acting_mode == 'value-mixture')

    s = true_env.reset()
    s = (s[0], s[1])
    r_wall = getattr(true_env, 'r_wall', None)
    goal = tuple(true_env.goal)

    total_return = 0.0
    collisions = 0
    bps_log = []
    step_records = []

    for t in range(int(max_steps)):
        # 1) action choice (action_count gets bumped here)
        a = meta.act(
            s, acting_mode=acting_mode,
            greedy=greedy, normalize=normalize,
        )

        # 2) bps without double-bumping action_count
        ac_snap = {k: list(v) for k, v in meta.action_count.items()}
        log_p = meta.log_prob_action(
            s, a, acting_mode=acting_mode,
            greedy=greedy, normalize=normalize,
        )
        meta.action_count = defaultdict(
            lambda: [0] * len(meta.action_space), ac_snap,
        )
        bps = -log_p / math.log(2.0)
        bps_log.append(bps)

        # 3) step the *true* environment
        s_next, r, done = true_env.step(a)
        s_next = (s_next[0], s_next[1])
        total_return += float(r)
        if r_wall is not None and float(r) == float(r_wall):
            collisions += 1

        # 4) self-supervised meta-agent update
        meta.observe_and_update(s, a, float(r), s_next)

        if record_per_step:
            step_records.append({
                't':                    t,
                's_x':                  s[0],
                's_y':                  s[1],
                'a':                    int(a),
                'r':                    float(r),
                'bps':                  float(bps),
                'belief_kl_to_uniform': _kl_to_uniform(meta.xi()),
            })

        s = s_next
        if done:
            break

    summary = {
        'steps':          len(bps_log),
        'reached_goal':   bool(s == goal),
        'total_return':   float(total_return),
        'collisions':     int(collisions),
        'bps_mean':       float(np.mean(bps_log)) if bps_log else float('nan'),
        'bps_sum':        float(np.sum(bps_log))  if bps_log else 0.0,
        'kl_to_uniform':  _kl_to_uniform(meta.xi()),
        'hit_cells':      len(getattr(meta, 'obstacle_hits', [])),
    }
    return summary, step_records


def _kl_to_uniform(belief):
    """KL(belief || Uniform_N).  Reads as 'how much info has been acquired'."""
    N = len(belief)
    if N == 0:
        return 0.0
    log_unif = math.log(1.0 / N)
    s = 0.0
    for b in belief:
        if b > 1e-12:
            s += b * (math.log(b) - log_unif)
    return float(s)
