# Algorithm 3-1. Q-learning (off-policy TD)
# 1: input: a policy that uses the action‐value function, π(a∣s,Q(s,a))
# 2: Initialize Q(s,a) = 0, for all s ∈ S, a ∈ A(s)
# 3: loop: for each episode
# 4: 		Initialize environment to provide s
# 5: 		do:
# 6: 			Choose a from s using π, breaking ties randomly
# 7: 			Take action, a, and observe r,s′
# 8: 			Q(s,a) = Q(s,a) + α[r + γ*argmax((a_s ∈ A(s), Q(s′,a_s))−Q(s,a)]
# 9: 			s = s′
#10: while s is not terminal

import os
import numpy as np
import random
from collections import defaultdict
from typing import Dict, Tuple, List, Any, Optional, Protocol
from tqdm import tqdm 
import pandas as pd

# --- 1) Define the GridWorld environment (deterministic) ---

use_horizontal_envs = True

from GridWorldCanonicalBenchmarkSuite import build_canonical_suite

# --- 2) ε-greedy behavior policy based on Q ---
def choose_action_epsilon_greedy(Q, state, nA, epsilon, action_count=None):
    """Select action using ε-greedy over Q[state]."""
    if random.random() < epsilon:
        return random.randrange(nA)
    else:
        # break ties randomly among best actions
        q_vals = Q[state]

        if action_count is not None:
            q_vals = np.array([qv*(1-np.sign(qv)*action_count[state][i]/100) for i, qv in enumerate(q_vals)])

        max_val = np.max(q_vals)
        # break ties randomly
        # best_actions = [a for a, q in enumerate(q_vals) if q == max_val]
        best_actions = np.flatnonzero(np.isclose(q_vals, max_val)).tolist()        
        action = random.choice(best_actions)

        if action_count is not None:
            action_count[state][action] += 1

        return action

def fake_tqdm(iter, desc):
    return iter

# --- 3) Q-learning off-policy TD control ---
def q_learning(env, num_episodes, alpha=0.5, 
               gamma=1.0,  
               epsilon_start=0.1,
               epsilon_end=0.05,
               prev_Q=None, 
               start_state=None,
               max_steps=-1,
               count_actions=False,
               visited_states = None,
               use_tqdm = True
               ):
    """
    Off-policy Q-learning:
    Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') - Q(s,a)].
    Behavior policy is ε-greedy on Q.
    """

    def eps_by_episode(ep):
        # linear decay
        frac = min(1.0, ep / max(1, epsilon_decay_steps))
        return epsilon_start + frac * (epsilon_end - epsilon_start)
    
    epsilon_decay_steps = num_episodes // 2

    nA = len(env.all_actions())    
    action_count = None

    # initialize Q(s,a)=0 for all state-action pairs
    if prev_Q is None:
        Q = defaultdict(lambda: np.zeros(nA))
    else:
        Q = prev_Q

    max_td_error_at_start = 0.0

    if use_tqdm:
        tqdm_iter = tqdm
    else:
        tqdm_iter = fake_tqdm

    rows = []
    for episode in tqdm_iter(range(num_episodes), desc="Episodes"):

        if count_actions:
            action_count=defaultdict(lambda: [0]*nA)

        epsilon = eps_by_episode(episode)

        start_state = env.reset(start_state=start_state)
        state = start_state

        done = False
        row = []
        t = -1
        while not done:
            t += 1

            if visited_states is not None:
                visited_states[state] += 1

            # --- Real interaction step ---
            action = choose_action_epsilon_greedy(Q, state, nA, epsilon, action_count)
            next_state, reward, done = env.step_from(state, action)

            # TD target uses the max over next-state actions
            best_next_q = np.max(Q[next_state])
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state][action]

            # Q-learning update
            Q[state][action] += alpha * td_error

            row.append({"episode":episode,"t":t,"obs":state,"act":action,"rew":reward})

            state = next_state

            # Track max delta at this state
            if abs(td_error) > max_td_error_at_start and state == start_state:
                max_td_error_at_start = abs(td_error)

            if max_steps > 0 and t > max_steps:
                break

        if done:
            rows += row[-1:]
        else:
            break

    df=pd.DataFrame(rows if done else [])

    return Q, df, max_td_error_at_start


# --- 4) Extract greedy policy and best trajectory ---
def extract_greedy_policy(env, Q):
    """Derive deterministic policy π(s)=argmax_a Q(s,a)."""
    policy = {}
    for s in env.all_states():
        if s == env.goal:
            policy[s] = None
        else:
            # policy[s] = int(np.argmax(Q[s]))
            # if np.std(Q[s]) == 0:
            #     policy[s] = random.randint(0, len(Q[s])-1)
            # else:
                q_vals = Q[s]
                # policy[s] = int(np.argmax(q_vals))
                max_val = np.max(q_vals)
                # break ties randomly
                # best_actions = [a for a, q in enumerate(q_vals) if q == max_val]
                best_actions = np.flatnonzero(np.isclose(q_vals, max_val)).tolist()
                policy[s] = random.choice(best_actions)                
    return policy

def softmax_row(z, temperature=1.0):
    z = z / max(temperature, 1e-6) # / np.sum(z)
    m = np.max(z)
    e = np.exp(np.clip(z - m, -50, 50))
    return e / (np.sum(e) + 1e-12)

def extract_stochastic_policy(env, Q, temperature=1.0):
    """Derive deterministic policy π(s)=softmax(Q(s))."""
    policy = {}
    for s in env.all_states():
        if s == env.goal or env.forbidden(s) or s not in Q:
            policy[s] = None
        else:
            # policy[s] = int(np.argmax(Q[s]))
            q_vals = Q[s]
            if np.std(q_vals) == 0:
                policy[s] = np.array([1/env.action_size]*env.action_size)
            else:
                policy[s] = softmax_row(q_vals, temperature)            
    return policy

def get_trajectory(env, policy, ep=-1, greedy=False, start_state=None, max_steps=100):
    """Follow the policy from start to goal, recording states."""
    state = env.reset(start_state=start_state)  
    
    border_end0 = env.border_end  
    # env.border_end = greedy

    traj = [(state[0], state[1])]
    rows = []
    for t in range(max_steps):   # cap length to avoid infinite loops
        a = policy[state] # [(state[0], state[1])]
        if a is None:
            break
        else:
            if (isinstance(a, np.ndarray)):
                if greedy:
                    max_val = np.max(a)
                    # break ties randomly
                    # best_actions = [a for a, q in enumerate(q_vals) if q == max_val]
                    best_actions = np.flatnonzero(np.isclose(a, max_val)).tolist()
                    a = random.choice(best_actions)                                      
                else:
                    a = np.random.choice(env.action_size, p=a)
        state1, reward, done = env.step_from(state, a)
        traj.append((state1[0], state1[1]))

        rows.append({"episode":ep, "t":t,"obs":state,"act":a,"rew":reward})        
        state = state1

        if done:
            break

    env.border_end = border_end0
    return traj, rows


def extract_learned_trajectories(env, policy, greedy=False, force_success=False, force_small=False, force_from_start=False):
    ep = -1
    final_rows = []

    tot_start_states = int(env.width*env.height*(multiple_starts_perc or default_multiple_starts_perc))
    max_tries = max(10, int(0.2*tot_start_states))

    if force_from_start:
        start_states = [env.start] * tot_start_states
    else:
        start_states = env.all_states()

    for s in tqdm(start_states):
        if not env.forbidden(s):
            ep += 1
            try_num = 1

            while try_num <= max_tries:
                # Trace out the best trajectory from start to goal
                best_traj, row = get_trajectory(env, policy, ep=ep, greedy=greedy, start_state=s, 
                                                            max_steps=5*(env.width+env.height) if force_small else 5*(env.width*env.height))
                if best_traj[-1] == env.goal or not force_success:
                    final_rows += row
                    break
                else:
                    try_num += 1

    df=pd.DataFrame(final_rows)

    return df

def load_qtable(agent_name: str, env: Any, data_dir: str, state_len: int, default_state_len: int, use_q_table_from_many_runs: bool = True ) -> pd.DataFrame:
    """
    Load trajectories for a given (agent, env).

    Assumes filenames like:
        f"{agent_name}_{env_name}_trajectory.csv"
    e.g. "sarsa_10x4_trajectory.csv"

    Expected columns: episode, t, obs, act, rew
    """

    env_name = env.name
    fname = f"{agent_name}_{env_name}{('-'+str(state_len)) if state_len != default_state_len else ''}_qtable{'_many' if use_q_table_from_many_runs else ''}.csv"
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):

        Q_df = pd.read_csv(fpath, index_col=0, header=[0])
        state_cols = list(Q_df.columns)[: -num_actions]

        Q_df['state'] = Q_df.apply(lambda x: tuple([int(x[c]) for c in state_cols]), axis=1)

        Q_df = Q_df.set_index('state', drop=True).drop(columns=state_cols)
        
        Q_df.columns = [int(a) for a in list(Q_df.columns)]

        action_space = [a for a in sorted(list(Q_df.columns))]

        Q_df['actions'] = Q_df.apply(lambda x: np.array([x[a] for a in action_space ]), axis=1)

        Q = Q_df.drop(columns=action_space).to_dict()['actions']
    else:
        Q = None
        action_space = None

    return Q, action_space


# --- 5) Demonstration on a 4×4 grid ---
if __name__ == "__main__":
    OUTPUT_DIR = './logs/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Create a 4×4 grid from (3,3) to (3,0) - Same grid as Figure 2-10 at page 52
    # env = GridWorld(width=4, height=4, start=(3,3), goal=(3,0), forbidden={(2,1), (3,1), (2,2), (3, 2)})

    # Create a 4×10 grid from (0,3) to (9,3) - Same grid as Figure 3-3 at page 68
    # env = GridWorld(width=10, height=4, start=(0,3), goal=(9,3), forbidden={(x, 3) for x in range(1,9)})
if __name__ == "__main__":

    random_maze_only = True # False # 
    lambda_cliff = 100.0 if random_maze_only else 100.0

    num_percs = 10 # 20
    perc0 = 0.0
    percN = 1.0      

    # num_percs = 40
    # perc0 = 0.5
    # percN = 0.75     
            
    nested_grids = False # True # 

    if random_maze_only:
        sizes = [(20,10)] 
        sizes = [(40,25)] 
        sizes = [(10, 5), (20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
        sizes = [(20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
        sizes = [(40, 20), (60,30), (80, 40), (100, 50)]
        sizes = [(60,30), (80, 40), (100, 50)]
        sizes = [(80, 40), (100, 50)]
        sizes = [(100, 50)]
        # sizes = [(10, 5), (20, 10), (40, 20), (60,30)]
        # sizes = [(10, 5)]
        # sizes = [(20, 10)]
        sizes = [(40, 20)]
        # sizes = [(60,30)]
        # sizes = [(80, 40)]
        # sizes = [(100, 50)]            

        
        random_maze_str = f'_rmaze{lambda_cliff:.1f}'
    else:
        random_maze_str = ''
    
        sizes = [(10, 4), (20, 10), (40, 20)]
        sizes = [(10,5)] 
        sizes = [(20,10)] 
        # sizes = [(40,20)] 
        # sizes = [(60,30)] 

    suite = build_canonical_suite(sizes, random_maze_only=random_maze_only, lambda_cliff=lambda_cliff, seed=123,
                                    num_percs=num_percs, nested_grids=nested_grids, perc0=perc0, percN=percN)

    envs = { env_name: env for (fam, env_name, i), env in suite.items() }

    N = len(envs)//len(sizes)

    # print_rewards = False
    # # Print one example per family at smallest size
    # printed = set()
    # for (fam, env_name, i), env in suite.items():
    #     if True or (env_name.endswith("10x4") and fam not in printed):
    #         print(f"=== {fam} / {env_name} ===")
    #         env.print_grid(print_rewards=print_rewards)
    #         printed.add(fam)

    tot_exp = 100

    temperature = 0.3
    alpha=0.1
    gamma=0.99 # 1.0
    
    agent_name = "qlearning"
    default_state_len = 2

    just_save_Q_table = True
    learn_from_multiple_starts = False # False means all possible starts

    default_multiple_starts_perc = 0.2 #  if learn_from_multiple_starts else 0.0
    if learn_from_multiple_starts:
        multiple_starts_perc_list = (np.array(list(range(1,11)))/10).tolist()
    else:
        multiple_starts_perc_list = [0.0]

    build_trajectories_from_saved_Q_table = True # False # 
    
    force_successfull_trajectories = True
    force_small_trajectories = False # True
    extract_trajectories_from_start_only = True

    use_greedy_policy_to_build_trajectories_from_saved_Q_table = True # False # 

    episode_mult = 1 # 2 # 
    
    for multiple_starts_perc in multiple_starts_perc_list:

        print()
        print("############################################################")
        print(f"################ multiple_starts_perc = {multiple_starts_perc} ################")
        print("############################################################")
        print()

        henv_str = 'h' if use_horizontal_envs else ''
        alpha_str = f'alpha{alpha}'
        from_start_str = 'fs' if extract_trajectories_from_start_only else ''
        multiple_starts_perc_str = f'{multiple_starts_perc:.2f}' if multiple_starts_perc > 0 else ''
        if multiple_starts_perc > 0:
            out_sub_dir = ('/greedy' if use_greedy_policy_to_build_trajectories_from_saved_Q_table else 
                        '/small_traj' if force_small_trajectories else
                        '/succ_traj' if force_successfull_trajectories else
                        '')
            out_sub_dir += multiple_starts_perc_str + from_start_str + alpha_str + henv_str

            qtable_sub_dir = f"/qtable{'_many' if multiple_starts_perc > 0 else ''}"
            qtable_sub_dir += multiple_starts_perc_str + alpha_str + henv_str
        else:
            out_sub_dir = "/traj_self_superv"
            qtable_sub_dir = "/qtable_self_superv"

        random_maze_str += f"x{episode_mult}" if episode_mult > 1 else ""
        
        out_sub_dir += random_maze_str + f"-N={N}" + ("-ng" if nested_grids else "-nn")
        qtable_sub_dir += random_maze_str + f"-N={N}" + ("-ng" if nested_grids else "-nn")

        if not os.path.exists(f"./logs{out_sub_dir}"):
            os.makedirs(f"./logs{out_sub_dir}")

        if not os.path.exists(f"./logs{qtable_sub_dir}"):
            os.makedirs(f"./logs{qtable_sub_dir}")

        succ_small_str = ''
        succ_small_str += '_succ' if force_successfull_trajectories else ''
        succ_small_str += '_small' if force_small_trajectories else ''
        succ_small_str += ('_greedy' if use_greedy_policy_to_build_trajectories_from_saved_Q_table else 
                            f'_stoch{temperature:.1f}')

        if build_trajectories_from_saved_Q_table:

            for (env, episodes) in list(envs.values())[:]:

                print(f'Building trajectories in Env: {env.name}!\n')

                env_name = env.name # f'{env.width}x{env.height}'
                state_len = len(env.reset())
                num_actions = len(env.all_actions())

                tot_start_states = int(env.width*env.height*multiple_starts_perc)
                use_q_table_from_many_runs = tot_start_states > 0
                Q, action_space  = load_qtable(agent_name, env, 
                                            data_dir=f"./logs{qtable_sub_dir}", 
                                            use_q_table_from_many_runs=use_q_table_from_many_runs, 
                                            state_len=state_len, default_state_len=default_state_len)

                if Q is None:
                    print(f'Q table not found for Env: {env.name}!\n')
                else:
                    demo_path2=f"./logs{out_sub_dir}/{agent_name}_{env.name}{succ_small_str}_traj.csv"

                    if os.path.exists(demo_path2):
                        print(f'\n{demo_path2} already exists! Skipping to next env size.\n')
                        continue   

                    df2 = pd.DataFrame(columns=["dummy"], index=[0, 1])
                    df2.to_csv(demo_path2,index=False)

                    # Derive greedy (optimal) policy from Q
                    policy = (Q if use_greedy_policy_to_build_trajectories_from_saved_Q_table else
                                         extract_stochastic_policy(env, Q, temperature))

                    df2 = extract_learned_trajectories(env, policy, 
                                                    greedy=use_greedy_policy_to_build_trajectories_from_saved_Q_table,
                                                    force_success=force_successfull_trajectories, 
                                                    force_small=force_small_trajectories,
                                                    force_from_start=extract_trajectories_from_start_only)
                    df2.to_csv(demo_path2,index=False)
        else:

            for (env, episodes) in envs.values():
                env_name = env.name # f'{env.width}x{env.height}'           

                tot_start_states = int(env.width*env.height*multiple_starts_perc)
                num_start_states = max(1, tot_start_states)
                start_state_num = 0
                max_tries = max(10, int(0.2*tot_start_states))

                Q = None
                last_fail = None
                try_num = 1
                while num_start_states > 0:

                    print(f'Learning Env with Q-learning: {env.name} from start #{start_state_num+1}/{tot_start_states}!\n')

                    if just_save_Q_table:
                        demo_path=f"./logs{qtable_sub_dir}/{agent_name}_{env.name}_qtable{'_many' if tot_start_states > 0 else ''}.csv"
                    else:
                        demo_path=f"./logs{out_sub_dir}/{agent_name}_{env.name}{succ_small_str}_traj.csv"

                    if os.path.exists(demo_path) and start_state_num == 0:
                        print(f'\n{demo_path} already exists! Skipping to next env size.\n')
                        break         
                    
                    if num_start_states > 1:
                        start_state = env.reset(random_start = start_state_num % 2 > 0, start_state=last_fail)
                    else:
                        start_state = env.reset()

                    delta_epsilon = (try_num-1)/100

                    Q, df = q_learning(env,
                                num_episodes=episodes,
                                alpha=alpha,
                                gamma=gamma,
                                epsilon_start=0.1+delta_epsilon,
                                epsilon_end=0.05,
                                prev_Q=Q,
                                start_state=start_state)

                    # Derive greedy (optimal) policy from Q
                    optimal_policy = extract_greedy_policy(env, Q)

                    if not just_save_Q_table:               
                        df.to_csv(demo_path,index=False)

                        df = extract_learned_trajectories(env, optimal_policy,
                                                    greedy=use_greedy_policy_to_build_trajectories_from_saved_Q_table,
                                                    force_success=force_successfull_trajectories, 
                                                    force_small=force_small_trajectories,
                                                    force_from_start=extract_trajectories_from_start_only)
                        demo_path2=f"./logs{out_sub_dir}/{agent_name}_{env.name}{succ_small_str}_traj.csv"
                        df.to_csv(demo_path2,index=False)

                    # Trace out the best trajectory under the learned policy
                    best_traj, _ = get_trajectory(env, optimal_policy, max_steps=2*(env.width*env.height), start_state=start_state)

                    print("Grid World schematics:")
                    env.print_grid()

                    print("Optimal State-Value Function Q:")
                    for j in range(env.height):
                        print([ round(max(Q[(i,j)]),1) for i in range(env.width) ])

                    print("\nOptimal Policy (0=↑,1=→,2=↓,3=←; None at goal):")
                    for j in range(env.height):
                        print([ round(np.argmax(Q[(i,j)]),1) for i in range(env.width) ])
            
                    print(f"\nBest Trajectory from Start to Goal len={len(best_traj)}:")
                    print(best_traj)

                    if best_traj[-1] == env.goal:
                        Q_df = pd.DataFrame(Q).T
                        action_cols = list(Q_df.columns)
                        Q_df = Q_df.reset_index()
                        Q_df.columns = ['x', 'y'] + action_cols
                        Q_path=f"./logs{qtable_sub_dir}/{agent_name}_{env.name}_qtable{'_many' if tot_start_states > 0 else ''}.csv"
                        Q_df.to_csv(Q_path, index=True, header=True)

                        start_state_num += 1
                        num_start_states -= 1
                        last_fail = None
                        try_num = 1
                    else:
                        last_fail = best_traj[-1]
                        try_num += 1
                        if try_num <= max_tries:
                            print(f'best_traj[-1]: {best_traj[-1]} != env.goal: {env.goal}!\nRetying to learn ({try_num}/{max_tries})!!!')   
                        else:
                            try_num = 1
                        if not just_save_Q_table:               
                            os.remove(demo_path)
                            os.remove(demo_path2)


