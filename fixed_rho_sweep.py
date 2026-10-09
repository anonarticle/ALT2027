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

from GridWorldCanonicalBenchmarkSuite import RandomMazeFamily, RandomScatterFamily, SizeSpec

from MultiAgentMainRecupE import load_qtable
from fixed_rho_sweep_analyze import analyze_results
from fixed_rho_sweep_helpers import run_vm_pms_rollouts, train_qtables_for_envs
from navigational_coherence_new import (
    _identify_consensus_free_space,
    compute_q_disagreement,
    compute_coherence_score,
)

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

def build_fixed_rho_family(size, rho, N, mode, master_seed, 
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

        # suite = build_canonical_suite(
        #     [size],
        #     random_maze_only=True,
        #     lambda_cliff=lambda_cliff,
        #     seed=seed_i,
        #     num_percs=10,        # one density level per call
        #     nested_grids=False,
        #     perc0=rho,
        #     percN=rho, verbose=False,
        #     **kwargs,
        # )
        # # build_canonical_suite returns dict; pull the single env out
        # (env, n_episodes), = [(e, ne) for (_, _, _), (e, ne) in suite.items()]

        env = fam.make(
            size=size_spec, scale=1, walls_perc=rho,
            seed=seed_i, obstacle_seed=hash((master_seed,i)) & 0x7FFFFFFF, verbose=False,
        )

        # Make the cache key unique per (mode, master_seed, i).
        env.name = f'{env.name}-{mode}-ms{master_seed}-i{i}'
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


# ---------------------------------------------------------------
# 3. The sweep.
# ---------------------------------------------------------------
def run_sweep(
    rho_grid       = [0.10, 0.20, 0.30],
    N_grid         = [1, 2, 3, 5, 8, 13], # , 21, 34],
    sizes          = [(40, 20)], # , (80, 40), (100, 50)],
    modes          = ['iid', 'shared_skel'],
    n_seeds        = 30,
    run_rollouts   = False,
    output_csv     = './logs/fixed_rho_sweep/fixed_rho_sweep.csv',
    qtable_root    = './logs/fixed_rho_sweep/qtable_self_superv',
    agent_name     = 'qlearning',
    lambda_cliff   = 100.0,
    grid_family    = 'maze',          # 'maze' | 'scatter'
    just_train_qtables = False,
):
    rows = []
    cells = list(product(sizes, modes, rho_grid, N_grid, range(n_seeds)))

    for size, mode, rho, N, seed in tqdm(cells, desc='sweep'):
        # Build the family.  For each realisation we draw a fresh
        # master_seed so that within a (size, mode, rho, N) cell the
        # n_seeds replications are independent.
        master_seed = hash((size, mode, rho, N, seed)) & 0xFFFFFFFF

        # 1. Build family
        envs = build_fixed_rho_family(
            size=size, rho=rho, N=N,
            mode=mode, master_seed=master_seed,
            lambda_cliff=lambda_cliff, 
            grid_family=grid_family,
        )

        # Q-tables: assumed to have been pre-trained per env.name.
        # If not, train here once per env and cache to disk.
        # The Q-table directory naming will need to reflect the
        # fixed-rho regime; pick a convention like:
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
            **geom,
            **roll,
        })

        # Periodic checkpoint
        if len(rows) % 100 == 0:
            pd.DataFrame(rows).to_csv(output_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode in zip(axes, ['iid', 'shared_skel']):
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
    g = sub.groupby(['mode', 'F_bin'])['fi_holds'].agg(['mean', 'count']).reset_index()
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
    # sizes = [(40, 20)]
    # sizes = [(60,30)]
    # sizes = [(80, 40)]
    # sizes = [(100, 50)]    

    # Sanity test!!!
    envs = build_fixed_rho_family(size=max(sizes), rho=0.3, N=5,
                                mode='iid', master_seed=12345)
    for i, e in enumerate(envs):
        walls = sorted(getattr(e, 'walls', getattr(e, 'obstacles', set())))
        print(f'env {i}: name={e.name}  |walls|={len(walls)}  hash={hash(tuple(walls)) & 0xFFFF:04x}')

    print()

    just_train_qtables = True # False # 

    out_dir = f'./logs/fixed_rho_sweep-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_csv     = out_dir+f'fixed_rho_sweep-{sizes[0][0]}x{sizes[0][1]}-{sizes[-1][0]}x{sizes[-1][1]}-S={len(sizes)}.csv'

    if not os.path.exists(output_csv):
        df = run_sweep( rho_grid       = [0.10, 0.20, 0.30, 0.40],
                        N_grid         = [1, 2, 3, 5, 8, 13], # , 21, 34],
                        sizes          = sizes, # [(40, 20)], # , (80, 40), (100, 50)],
                        modes          = ['iid'], # , 'shared_skel'],
                        n_seeds        = 30,
                        run_rollouts   = False,
                        output_csv     = output_csv, # './logs/fixed_rho_sweep/fixed_rho_sweep.csv',
                        qtable_root    = out_dir+'qtable_self_superv',
                        agent_name     = 'qlearning',
                        lambda_cliff   = 100.0,
                        grid_family    = 'maze',          # 'maze' | 'scatter'
                        just_train_qtables = just_train_qtables,
        )
                                
    else:
        df = pd.read_csv(output_csv).reset_index()

    if not just_train_qtables:
        grouped = aggregate_success_rate(df)

        for size in sizes:
            plot_A_success_vs_N(grouped, size_filter=f'{size[0]}x{size[1]}', fig_name=out_dir+f'fixed_rho_sweep-A_success_vs_N-{size[0]}x{size[1]}.png')

            plot_B_collapse(df, size_filter=f'{size[0]}x{size[1]}', fig_name=out_dir+f'fixed_rho_sweep-B_collapse-{size[0]}x{size[1]}.png')

        analyze_results(output_csv)

    print()