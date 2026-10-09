
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import itertools
import os
from typing import Dict, Tuple, List, Any, Optional, Protocol, get_args, get_origin
import math
import time
import random
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from Q_learning_Agent_plus_plus import q_learning, get_trajectory

# from NaiveMultiEnvMetaAgent import MetaAgent
# from MetaAgent2OGSelfSup import MetaAgent2OGSelfSup as MetaAgent
from MetaAgent2NIGSelfSup import MetaAgent2NIGSelfSup as MetaAgent
from MetaAgent2NIGSelfSup import value_mixture_action_temperature

from GridWorldCanonicalBenchmarkSuite import build_canonical_suite

State = Tuple[int, int]          # (x, y)
Action = int                     # 0,1,2,3

# Convention for actions:
# 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
# ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
#     0: (0, -1),
#     1: (1, 0),
#     2: (0, 1),
#     3: (-1, 0),
# }

ACTION_DELTAS = [(0,-1),(1,0),(0,1),(-1,0)]

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def normalize(probs: List[float], eps: float = 1e-12) -> List[float]:
    s = sum(probs)
    if s < eps:
        # fallback to uniform if degenerate
        return [1.0 / len(probs)] * len(probs)
    return [p / s for p in probs]


def doSeed(seed=None, verbose=True):
    if seed is None:
        seed = int(time.time()*1000) % (2**32) 
    
    if verbose: print(seed)
    np.random.seed(seed)
    random.seed(seed)

    return seed

class Environment(Protocol):
    """
    Minimal interface expected by MetaAgent.
    You can adapt your own Environment class to this.
    """
    width: int
    height: int
    start: State
    max_steps: int

    def reset(self) -> State: ...
    # def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]: ...
    def step(self, action: Action) -> Tuple[State, float, bool]: ...
    def step_from(self, s: State, action: Action) -> Tuple[State, float, bool]: ...


class GridWorldEnv:
    """
    Simple reference implementation (optional). Obstacles are impassable.
    Reward: -1 for successful move to empty cell,
            -10 for attempting move into obstacle (no move),
            0 for reaching goal (customizable).
    """
    def __init__(
        self,
        width: int,
        height: int,
        obstacles: set[State],
        start_state: State,
        goal_state: State,
        max_steps: int = 200,
        step_cost: float = -1.0,
        obstacle_cost: float = -10.0,
        goal_reward: float = 0.0,
        border_end: bool = False,
    ):
        self.name = "Env"        
        self.width = width
        self.height = height
        self.walls = set(obstacles)
        self.start = start_state
        self.goal = goal_state
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.obstacle_cost = obstacle_cost
        self.goal_reward = goal_reward
        self.border_end = border_end

        # actions: 0=up, 1=right, 2=down, 3=left  (0=↑,1=→,2=↓,3=←)
        self.action_deltas = ACTION_DELTAS # [(0,-1),(1,0),(0,1),(-1,0)]
        # self.action_arrows = {0:'↑',1:'→',2:'↓',3:'←'}
        self.action_arrows = {'up':'↑','right':'→','down':'↓','left':'←'}
        self.action_size = len(self.action_deltas)
        self.action_space = list(range(self.action_size))

        self._s = start_state
        self._t = 0

    def update_r_wall(self, wall_reward_like_cliff):
        # self.r_wall = self.r_cliff if wall_reward_like_cliff else self.r_step
        pass

    def reset(self, random_start=False, start_state=None) -> State:
        if start_state is None:
            if random_start:
                while True:
                    x = random.randint(0,self.width-1)
                    y = random.randint(0,self.height-1)
                    s = (x, y)
                    if s != self.goal and s not in self.walls:
                        break
                self._s = (x, y)
            else:
                self._s = self.start
        else:
            self._s = start_state

        self._t = 0
        return self._s
    
    def forbidden(self, s: State):
        return s in self.walls
                     
    def step(self, action: Action) -> Tuple[State, float, bool]: # , Dict[str, Any]]:
        """
        Deterministic transition: from 'self.state' take 'action', 
        return (next_state, reward, done).
        """
        return self.step_from(self._s, action)

    def step_from(self, s, action: Action) -> Tuple[State, float, bool]: # , Dict[str, Any]]:
        """
        Deterministic transition: from 'state' take 'action', 
        return (next_state, reward, done).
        """
        self._t += 1
        self._s = s
        x, y = self._s
        dx, dy = ACTION_DELTAS[action]
        # nx = clamp(x + dx, 0, self.width - 1)
        # ny = clamp(y + dy, 0, self.height - 1)
        # ns = (nx, ny)

        if (0 <= x + dx < self.width and
            0 <= y + dy < self.height):
            ns = (x + dx, y + dy)
            r = self.step_cost # obstacle_cost
        else:
            ns = (x, y)
            r = self.step_cost
            done = self.border_end

        info: Dict[str, Any] = {"attempted_next_state": ns}

        # Hit obstacle -> no move
        if ns in self.walls:
            ns = self._s
            r = self.obstacle_cost
            done = False
            self._s = ns
            if self._t >= self.max_steps:
                done = True
            return ns, r, done # , info

        # Move to empty cell
        self._s = ns

        # Goal?
        if ns == self.goal:
            r = self.goal_reward
            done = True
        else:            
            done = False

        if self._t >= self.max_steps:
            done = True

        return ns, r, done # , info

    def all_states(self):
        """List of all (i,j) states."""
        return [(i, j) for i in range(self.width) for j in range(self.height)]

    def all_actions(self):
        """List of action indices."""
        return list(range(len(self.action_deltas)))
    
    def print_grid(self, trajectory=None, print_rewards=None):
        """
        Print the grid blueprint: S for start, G for goal, · for empty.
        """
        print(self.name)
        nsteps = 0 if trajectory is None else len(trajectory)
        for j in range(self.height):
            row = []
            for i in range(self.width):
                if (i, j) == self.start:
                    row.append("S")
                elif (i, j) == self.goal:
                    row.append("G")
                elif (i, j) in self.walls:
                    row.append("X")
                else:
                    if trajectory is not None and (i, j) in trajectory:
                        pos = trajectory[::-1].index((i, j))
                        pos = nsteps - pos - 1
                        if pos < nsteps-1:
                            ni, nj = trajectory[pos+1]
                        else:
                            if trajectory[-1] == self.goal:
                                ni, nj = self.goal
                            else:
                                ni, nj = i, j
                        # print( ni, nj)
                        if ni > i:
                            row.append(self.action_arrows['right'])        
                        elif ni < i:
                            row.append(self.action_arrows['left'])        
                        elif nj < j:
                            row.append(self.action_arrows['up'])        
                        elif nj > j:
                            row.append(self.action_arrows['down'])   
                        else:
                            row.append("·")
                    else:
                        row.append("·")
            print(" ".join(row))
        print()  # blank line for spacing

def load_qtable(agent_name: str, env_name: str, data_dir: str, state_len: int, default_state_len: int , suffix = "") -> pd.DataFrame:
    """
    Load trajectories for a given (agent, env).

    Assumes filenames like:
        f"{agent_name}_{env_name}_trajectory.csv"
    e.g. "sarsa_10x4_trajectory.csv"

    Expected columns: episode, t, obs, act, rew
    """

    fname = f"{agent_name}_{env_name}{('-'+str(state_len)) if state_len != default_state_len else ''}_qtable{suffix}.csv"
    fpath = os.path.join(data_dir, fname)
    Q_df = pd.read_csv(fpath, index_col=0, header=[0])

    state_cols = list(Q_df.columns)[: -len(ACTION_DELTAS)]
    Q_df['state'] = Q_df.apply(lambda x: tuple([int(x[c]) for c in state_cols]), axis=1)

    Q_df = Q_df.set_index('state', drop=True).drop(columns=state_cols)

    Q_df.columns = [int(a) for a in list(Q_df.columns)]

    action_space = [a for a in sorted(list(Q_df.columns))]

    Q_df['actions'] = Q_df.apply(lambda x: np.array([x[a] for a in action_space ]), axis=1)

    Q = defaultdict(lambda: np.zeros(len(action_space)), Q_df.drop(columns=action_space).to_dict()['actions'])

    return Q, action_space



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

def extract_meta_agent_trajectories(env, meta, 
                                    mode="policy-mixture",  #  "policy-mixture"/"value-mixture"/"thompson"
                                    greedy= False,
                                    force_success=False, 
                                    force_small=False,
                                    force_from_start = False,
                                    normalize=False,
                                    multiple_starts_perc=0.2):
    ep = -1
    final_rows = []

    Q = meta.q_tables[-1]
    optimal_policy = extract_greedy_policy(env, Q)
    best_traj2, _ = get_trajectory(env, optimal_policy, max_steps=2*(env.width*env.height), start_state=env.start)

    tot_start_states = int(env.width*env.height*multiple_starts_perc)
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
                row , best_traj = meta.run_episode(env, 
                                                  acting_mode=mode, #  "value-mixture"/"thompson"
                                                  episode=ep, 
                                                  start_state=s, 
                                                  max_steps=5*(env.width+env.height) if force_small else env.width*env.height*5, 
                                                  greedy=greedy,
                                                  normalize=normalize)

                if best_traj[-1] == env.goal or not force_success: # or greedy: @@@@@@@@@@@@@
                    final_rows += row
                    break
                else:
                    try_num += 1

    df=pd.DataFrame(final_rows)

    return df


def build_meta_agent_trajectories(envs_dic, meta, 
                                  mode="value-mixture",  #  "value-mixture"/"thompson"
                                  greedy= False,                                  
                                  data_dir="./", 
                                  force_success=False, 
                                  force_small=False, 
                                  force_save=False,
                                  force_from_start = False,
                                  normalize=False,
                                  multiple_starts_perc=0.2):

    for (env, _) in envs_dic.values():

        print(f'\nBuilding Meta Agent trajectories in Env: {env.name}!')
    
        env_name = env.name # f'{env.width}x{env.height}'
        # meta = env_meta_agents[env_name]
        
        # mode_str = "pm" if mode == "policy-mixture" else "vm" if mode == "value-mixture" else "th" if mode == "thompson" else "invalid_mode"
        # demo_path2=data_dir+f"/meta_agent_{env.width}x{env.height}_learned{'_succ' if force_success else ''}{'_small' if force_small else ''}{'_greedy' if greedy else ''}_traj_{tot_exp}_{mode_str}.csv"

        # num_learned_str = '_'+str(num_trajectories) if use_multiple_trajectories else '_learned' if use_learned_trajectories else ''
        succ_str = '_succ' if force_success else ''
        small_str = '_small' if force_small else ''
        greedy_str = '_greedy' if greedy else f'_stoch{value_mixture_action_temperature:.1f}'
        mode_str = "pm" if mode == "policy-mixture" else "vm" if mode == "value-mixture" else "th" if mode == "thompson" else "invalid_mode"
        if mode_str == 'vm':
            mode_str += f'{value_mixture_action_temperature}' + ('n' if normalize_q_values_in_value_mixture else '')      

        fname = f"meta_agent_{env_name}{succ_str}{small_str}{greedy_str}_traj_{mode_str}.csv"    

        demo_path2 = os.path.join(data_dir, fname)
        
        if force_save or not os.path.exists(demo_path2):
                
            df2 = pd.DataFrame(columns=["dummy"], index=[0, 1])
            df2.to_csv(demo_path2,index=False)
                            
            df2 = extract_meta_agent_trajectories(env, meta, 
                                                mode=mode,  #  "policy-mixture"/"value-mixture"/"thompson"
                                                greedy=greedy,                                              
                                                force_success=force_success, 
                                                force_small=force_small,
                                                force_from_start=force_from_start,
                                                normalize=normalize,
                                                multiple_starts_perc=multiple_starts_perc)
            
            print(f'Saving Meta Agent trajectories to: {demo_path2}!')
            df2.to_csv(demo_path2,index=False)

def euclidean_distance(tup1, tup2):
    # Assumes tuples have the same number of elements
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(tup1, tup2)]))
    return distance

def manhattan_distance(tup1, tup2):
    # Assumes tuples have the same number of elements
    distance = sum([abs(a - b) for a, b in zip(tup1, tup2)])
    return distance

def stateQTableIsStable(env, q, 
                        num_episodes=1000, 
                        alpha=0.1,
                        gamma=0.99,
                        epsilon_start=0.1,
                        epsilon_end=0.05,
                        start_state=None,
                        max_steps=-1,
                        count_actions=False,
                        visited_states=None,
                        skip_threshold=0.005,
                        use_tqdm=True):
    pass
   
    q, df, max_td_error_at_start = q_learning(env,
        num_episodes=min(200, num_episodes // 10), 
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        prev_Q=q,
        start_state=start_state,
        max_steps=10*(env.width*env.height),
        count_actions=count_actions,
        visited_states=visited_states,
        use_tqdm=use_tqdm) 
                
    return max_td_error_at_start < skip_threshold, max_td_error_at_start


def trainQTable(env, qtable_sub_dir, suffix, num_episodes=2000):

    if True:

        if not os.path.exists(f"./logs{qtable_sub_dir}"):
            os.makedirs(f"./logs{qtable_sub_dir}")
            
        nA = len(env.all_actions())
        state_len = len(env.reset())
        env.update_r_wall(wall_reward_like_cliff)

        Q_path=f"./logs{qtable_sub_dir}/{agent_name}_{env.name}{('-'+str(state_len)) if save_full_state_info and state_len != default_state_len else ''}_qtable{suffix}.csv" # "{'D' if sort_states_reversed else 'C'}.csv"
        Q_mark_path=f"./logs{qtable_sub_dir}/{agent_name}_{env.name}{('-'+str(state_len)) if save_full_state_info and state_len != default_state_len else ''}_qtable{suffix}_mark.csv" # "{'D' if sort_states_reversed else 'C'}_mark.csv"
        Q_mark_path2 = Q_mark_path[:-4]+'2.csv'

        if (not os.path.exists(Q_path)) or os.path.exists(Q_mark_path):

            if not os.path.exists(Q_path):
                Q_df = pd.DataFrame(columns=["dummy"], index=[0, 1])
                Q_df.to_csv(Q_path,index=False)
                q = None
                last_good_q = None
            else: # add code to check if q is dummy!
                Q_df = pd.read_csv(Q_path)
                if "dummy" in Q_df.columns:
                    q = None
                    last_good_q = None
                else:
                    q, action_space = load_qtable(agent_name, env.name, data_dir=f"./logs{qtable_sub_dir}/", 
                                        state_len=state_len, default_state_len=default_state_len)       
                    last_good_q = defaultdict(lambda: np.zeros(len(ACTION_DELTAS)), dict(q))         
                
            done_states = set()
            if os.path.exists(Q_mark_path):
                os.rename(Q_mark_path, Q_mark_path2)
                mark_df = pd.read_csv(Q_mark_path2)
                mark_df['state'] = mark_df['state'].apply(lambda s: tuple([int(p) for p in s[1:-1].lower().split(",")]))
                done_states = set(mark_df['state'].values.tolist()[:-1])
            else:                    
                mark_df = pd.DataFrame(columns=["state"])
                mark_df.to_csv(Q_mark_path2,index=False)

            done_states.add(env.goal)

            max_distance = env.width + env.height
            min_num_visits = 10
            # num_episodes=num_episodes_list[nu]*int(episode_mult)
            if sort_states_reversed:
                sorted_states = sorted(env.all_states(), key=lambda x: sum([(a - b) ** 2 for a, b in zip(x, env.goal)]))[1:]
            else:
                sorted_states = sorted(env.all_states(), key=lambda x: sum([(a - b) ** 2 for a, b in zip(x, env.goal)]), reverse=True)[:-1]
                # sorted_states = env.all_states()     

            diagonal_states = [ s for i, s in enumerate([s for s in sorted_states if abs((s[0]+1)*env.height/env.width - (env.height-s[1])) <= 1]) if i%2 == 0 ]

            sorted_states = diagonal_states + random.sample([s for s in sorted_states if s not in diagonal_states], env.width//2)
            num_states = len(sorted_states) 
                               
            # sorted_states = [env.start]*num_states

            while len(done_states) < num_states:
            
                for k, (i, j) in enumerate(tqdm(sorted_states[:]+[env.start]*2, desc=env.name+suffix)):
                    start_state = (i, j)

                    state_dist = manhattan_distance(start_state, env.goal)
                    dist_frac = state_dist / max_distance
                    ep_perc = max(0.1, dist_frac)
                      
                    if start_state in done_states and k < num_states:
                        continue

                    if len(mark_df) == 0 or mark_df['state'].iloc[-1] != start_state:
                        mark_df.loc[len(mark_df), 'state'] = start_state
                        mark_df.to_csv(Q_mark_path2,index=False)

                    if start_state != env.goal and not env.forbidden(start_state):

                        while True:

                            q, df, _ = q_learning(env,
                                num_episodes=int(num_episodes*ep_perc), 
                                alpha=0.1,
                                gamma=1.0,
                                epsilon_start=0.1,
                                epsilon_end=0.05,
                                prev_Q=q,
                                start_state=start_state,
                                max_steps=-1,
                                count_actions=count_actions_to_break_loops,
                                visited_states=None,
                                use_tqdm=test_visited_states) # 1000*(env.width*env.height))
                                        
                            best_traj, _ = get_trajectory(env, q, greedy=True, max_steps=5*(env.width*env.height), start_state=start_state)
                            if best_traj[-1] == env.goal and len(df) > 0:
                                last_good_q = defaultdict(lambda: np.zeros(len(ACTION_DELTAS)), dict(q))
                                break
                            else:
                                bbb = 1
                                if last_good_q is not None:
                                    q = defaultdict(lambda: np.zeros(len(ACTION_DELTAS)), dict(last_good_q))
                                else:
                                    q = None
                                seed = doSeed(None, verbose=False) 

                        done_states.add(start_state)

                        Q_df = pd.DataFrame(q).T
                        action_cols = list(Q_df.columns)
                        Q_df = Q_df.reset_index()
                        if state_len > default_state_len and not save_full_state_info:                
                            state_cols = list(Q_df.columns)[:-len(action_cols)]
                            Q_df = Q_df.sort_values(state_cols).drop_duplicates(subset=state_cols[:default_state_len], keep='last')
                            Q_df = Q_df[list(Q_df.columns)[:default_state_len]+action_cols]
                            Q_df.columns = ['x', 'y'] + action_cols
                        else:
                            Q_df.columns = list(env.state_columns()) + action_cols
                        Q_df.to_csv(Q_path, index=True, header=True)
                    
                break
            
            if os.path.exists(Q_mark_path2):
                Q_mark_path3 = Q_mark_path[:-4]+'3.csv'
                os.rename(Q_mark_path2, Q_mark_path3)
                # os.remove(state_count_path)
                
    
# --------------------- Example usage ---------------------
use_simple_envs = False
wall_reward_like_cliff = True
multiple_starts_perc = 0.2


to_build_meta_agent_trajectories = False # True # 


test_visited_states = False

count_actions_to_break_loops = True

save_full_state_info = True

force_save = False # True
ma_force_successfull_trajectories = True
force_small_trajectories = False # True
extract_trajectories_from_start_only = True

normalize_q_values_in_value_mixture = True    

sort_states_reversed = True # False # 

episode_mult = 1 # 2 # 

agent_name = 'qlearning'

default_state_len = len(get_args(State))


if __name__ == "__main__":


    seed = None
    # seed = 2345467781
    seed = doSeed(seed)    

    num_percs = 10 # 20
    perc0 = 0.0
    percN = 1.0      

    # num_percs = 40
    # perc0 = 0.5
    # percN = 0.75     
            
    nested_grids = False # True # 

    if use_simple_envs:
        # Dummy example with 2 envs, same grid size, different obstacle placement.
        W, H = 6, 6
        start = (0, 0)
        goal = (5, 5)

        env1 = GridWorldEnv(W, H, obstacles={(2, 1), (2, 2), (2, 0), (2, 3)}, start_state=start, goal_state=goal)
        env2 = GridWorldEnv(W, H, obstacles={(1, 2), (2, 2), (3, 2), (0, 2)}, start_state=start, goal_state=goal)
        env3 = GridWorldEnv(W, H, obstacles={(1, 2), (2, 2), (1, 3), (2, 1)}, start_state=start, goal_state=goal)
        # env4 = GridWorldEnv(W, H, obstacles={(3, 5), (1, 4)}, start_state=start, goal_state=goal)

        envs = [env1, env2, env3] # , env4]
        num_episodes_list = [500, 500, 500] # , 500]
        for i, env in enumerate(envs):
            env.name = f"Env{i+1}"

        random_maze_str = ''
    else:
        random_maze_only = True # False # 
        lambda_cliff = 100.0 if random_maze_only else 100.0

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
            # sizes = [(40, 20)]
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

    created_qtables = False

    if to_build_meta_agent_trajectories and len(sizes) > 1:
        envs = []

    visited_states = {}
    all_action_space = set()
    qs = []
    for nu, env in enumerate(tqdm(envs[:])):

        n_planning_steps = max(20, int(0.4 * (env.width+env.height)))
        
        nA = len(env.all_actions())
        state_len = len(env.reset())
        env.update_r_wall(wall_reward_like_cliff)

        Q_path=f"./logs{qtable_sub_dir}/{agent_name}_{env.name}{('-'+str(state_len)) if save_full_state_info and state_len != default_state_len else ''}_qtable.csv" # "{'D' if sort_states_reversed else 'C'}.csv"
        Q_mark_path=f"./logs{qtable_sub_dir}/{agent_name}_{env.name}{('-'+str(state_len)) if save_full_state_info and state_len != default_state_len else ''}_qtable_mark.csv" # "{'D' if sort_states_reversed else 'C'}_mark.csv"
        Q_mark_path2 = Q_mark_path[:-4]+'2.csv'
        state_count_path = Q_mark_path[:-4]+'.pkl'
        if (not os.path.exists(Q_path)) or os.path.exists(Q_mark_path):

            if not os.path.exists(Q_path):
                Q_df = pd.DataFrame(columns=["dummy"], index=[0, 1])
                Q_df.to_csv(Q_path,index=False)
                q = None
                last_good_q = None
            else: # add code to check if q is dummy!
                Q_df = pd.read_csv(Q_path)
                if "dummy" in Q_df.columns:
                    q = None
                    last_good_q = None
                else:
                    q, action_space = load_qtable(agent_name, env.name, data_dir=f"./logs{qtable_sub_dir}/", 
                                        state_len=state_len, default_state_len=default_state_len)       
                    last_good_q = defaultdict(lambda: np.zeros(len(ACTION_DELTAS)), dict(q))         
                
            done_states = set()
            if os.path.exists(Q_mark_path):
                os.rename(Q_mark_path, Q_mark_path2)
                mark_df = pd.read_csv(Q_mark_path2)
                mark_df['state'] = mark_df['state'].apply(lambda s: tuple([int(p) for p in s[1:-1].lower().split(",")]))
                done_states = set(mark_df['state'].values.tolist()[:-1])
            else:                    
                mark_df = pd.DataFrame(columns=["state"])
                mark_df.to_csv(Q_mark_path2,index=False)

            done_states.add(env.goal)

            if test_visited_states and os.path.exists(state_count_path):
                with open(state_count_path, 'rb') as pkl_file:
                    visited_states = defaultdict(lambda : 0, pickle.load(pkl_file))

            else:
                visited_states = defaultdict(lambda : 0)

            max_distance = env.width + env.height
            min_num_visits = 10
            num_episodes=num_episodes_list[nu]*int(episode_mult)
            if sort_states_reversed:
                sorted_states = sorted(env.all_states(), key=lambda x: sum([(a - b) ** 2 for a, b in zip(x, env.goal)]))[1:]
            else:
                sorted_states = sorted(env.all_states(), key=lambda x: sum([(a - b) ** 2 for a, b in zip(x, env.goal)]), reverse=True)[:-1]
                # sorted_states = env.all_states()     

            num_states = len(sorted_states)
            num_enough_visited = 0
                               
            while len(done_states) < num_states:
            
                num_enough_visited = 0
                for k, (i, j) in enumerate(tqdm(sorted_states[:]+[env.start]*2, desc=env.name)):
                    start_state = (i, j)

                    state_dist = manhattan_distance(start_state, env.goal)
                    dist_frac = state_dist / max_distance
                    ep_perc = max(0.1, dist_frac)

                    if test_visited_states and (start_state in visited_states) and (start_state not in done_states) and start_state != env.start:
                        stateQTableIsStableRes, max_td_error_at_start = stateQTableIsStable(env, q, 
                                                num_episodes=int(num_episodes*ep_perc), 
                                                alpha=0.1,
                                                gamma=1.0,
                                                epsilon_start=0.1,
                                                epsilon_end=0.05,
                                                start_state=start_state,
                                                max_steps=-1,
                                                count_actions=count_actions_to_break_loops,
                                                visited_states=visited_states)
                        if stateQTableIsStableRes:
                            num_enough_visited += 1
                            done_states.add(start_state)
                            continue
                        
                    if start_state in done_states and k < num_states:
                        continue

                    if len(mark_df) == 0 or mark_df['state'].iloc[-1] != start_state:
                        mark_df.loc[len(mark_df), 'state'] = start_state
                        mark_df.to_csv(Q_mark_path2,index=False)

                    if start_state != env.goal and not env.forbidden(start_state):

                        while True:

                            q, df, _ = q_learning(env,
                                num_episodes=int(num_episodes*ep_perc), 
                                alpha=0.1,
                                gamma=1.0,
                                epsilon_start=0.1,
                                epsilon_end=0.05,
                                prev_Q=q,
                                start_state=start_state,
                                max_steps=-1,
                                count_actions=count_actions_to_break_loops,
                                visited_states=visited_states,
                                use_tqdm=test_visited_states) # 1000*(env.width*env.height))
                                        
                            best_traj, _ = get_trajectory(env, q, greedy=True, max_steps=5*(env.width*env.height), start_state=start_state)
                            if best_traj[-1] == env.goal and len(df) > 0:
                                last_good_q = defaultdict(lambda: np.zeros(len(ACTION_DELTAS)), dict(q))
                                break
                            else:
                                bbb = 1
                                if last_good_q is not None:
                                    q = defaultdict(lambda: np.zeros(len(ACTION_DELTAS)), dict(last_good_q))
                                else:
                                    q = None
                                seed = doSeed(None, verbose=False) 

                        done_states.add(start_state)

                        Q_df = pd.DataFrame(q).T
                        action_cols = list(Q_df.columns)
                        Q_df = Q_df.reset_index()
                        if state_len > default_state_len and not save_full_state_info:                
                            state_cols = list(Q_df.columns)[:-len(action_cols)]
                            Q_df = Q_df.sort_values(state_cols).drop_duplicates(subset=state_cols[:default_state_len], keep='last')
                            Q_df = Q_df[list(Q_df.columns)[:default_state_len]+action_cols]
                            Q_df.columns = ['x', 'y'] + action_cols
                        else:
                            Q_df.columns = list(env.state_columns()) + action_cols
                        Q_df.to_csv(Q_path, index=True, header=True)
                        created_qtables = True
                        action_space = action_cols
                    
                        if test_visited_states:
                            with open(state_count_path, 'wb') as pkl_file:
                                pickle.dump(dict(visited_states), pkl_file, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol for best performance                        

                break
            
            if os.path.exists(Q_mark_path2):
                Q_mark_path3 = Q_mark_path[:-4]+'3.csv'
                os.rename(Q_mark_path2, Q_mark_path3)
                # os.remove(state_count_path)
                
        # else:
        #     q, action_space = load_qtable(agent_name, env.name, data_dir=f"./logs{qtable_sub_dir}/", 
        #                                   state_len=state_len, default_state_len=default_state_len)

        # qs.append(q)
        # all_action_space.update(action_space)

    # for i, q in enumerate(qs):
    #     print(f"\nWorst State-Value Function Q: {envs[i].name}")
    #     for j in range(env.height):
    #         print([ round(min(q[(i,j)]),1) for i in range(env.width) ])

    if created_qtables or len(visited_states) > 0:
        quit()

    qs = []
    for nu, env in enumerate(tqdm(envs)):
        state_len = len(env.reset())
        env.update_r_wall(wall_reward_like_cliff)
        q, action_space = load_qtable(agent_name, env.name, data_dir=f"./logs{qtable_sub_dir}/", 
                                        state_len=state_len, default_state_len=default_state_len)
        qs.append(q)
        all_action_space.update(action_space)

    all_action_space = sorted(list(all_action_space))

    if to_build_meta_agent_trajectories:

        sizes0 = [(10, 5), (20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
        # sizes0 = [(80, 40), (100, 50)]
        sizes0 = [(100, 50)]
        sizes0 = [(40, 20)]
            
        for size in sizes0[:]:
            all_action_space = set()   

            sizes = [size]
            suite = build_canonical_suite(sizes, random_maze_only=random_maze_only, lambda_cliff=lambda_cliff, seed=123,
                                          num_percs=num_percs, nested_grids=nested_grids, perc0=perc0, percN=percN)

            envs_eps = [ (env, n_episodes) for (fam, env_name, i), (env, n_episodes) in suite.items() ]

            envs, num_episodes_list = zip(*envs_eps)
            envs_dic = { env_name: env for (fam, env_name, i), env in suite.items() }


            qs = []
            for nu, env in enumerate(tqdm(envs[:])):
                state_len = len(env.reset())
                env.update_r_wall(wall_reward_like_cliff)
                q, action_space = load_qtable(agent_name, env.name, data_dir=f"./logs{qtable_sub_dir}/", 
                                                state_len=state_len, default_state_len=default_state_len, suffix="")
                qs.append(q)
                all_action_space.update(action_space)

            all_action_space = sorted(list(all_action_space))

            # Meta-agent only needs an env "template" for width/height semantics
            meta = MetaAgent(env_template=envs[0], q_tables=qs, action_space=all_action_space, epsilon=0.0, use_belief_weighted_Q=False) # , unknown_rew_range = True)

            greedy=True
            build_meta_agent_trajectories(envs_dic, meta, 
                                    mode="thompson",  #  "value-mixture"/"thompson"
                                    greedy= greedy,                                  
                                    data_dir=traj_sub_dir, 
                                    force_success=ma_force_successfull_trajectories, 
                                    force_small=force_small_trajectories,
                                    force_save=force_save,
                                    force_from_start=extract_trajectories_from_start_only,
                                    normalize=normalize_q_values_in_value_mixture,
                                    multiple_starts_perc=0.2)
    else:
        # Meta-agent only needs an env "template" for width/height semantics
        meta = MetaAgent(env_template=envs[0], q_tables=qs, action_space=all_action_space, epsilon=0.0, use_belief_weighted_Q=False) # , unknown_rew_range = True)

        qs = []
        for nu, env in enumerate(tqdm(envs)):
            state_len = len(env.reset())
            env.update_r_wall(wall_reward_like_cliff)
            q, action_space = load_qtable(agent_name, env.name, data_dir=f"./logs{qtable_sub_dir}/", 
                                            state_len=state_len, default_state_len=default_state_len)
            qs.append(q)
            all_action_space.update(action_space)

        all_action_space = sorted(list(all_action_space))

        # Meta-agent only needs an env "template" for width/height semantics
        meta = MetaAgent(env_template=envs[0], q_tables=qs, action_space=all_action_space, epsilon=0.0, use_belief_weighted_Q=False) # , unknown_rew_range = True)

        for nu, env in enumerate(envs[:]):
            print(f"\nObstacle probabilities for {env.name}:")
            env.print_grid()        
            for j in range(env.height):
                print([ f"{meta.obstacle_prob[nu][j][i]:5.3f}" for i in range(env.width) ])

            succ_trajs = 0
            for ep in tqdm(range(1000)):
                if ep == 999:
                    e = 1111

                # Run an episode in env1 (true env sampled externally)
                traj, traj_states = meta.run_episode(env, 
                                                     greedy=True,
                                                     acting_mode="thompson",  #  "value-mixture"/"thompson"
                                                     max_steps=env.width*env.height * 5,
                                                     normalize=normalize_q_values_in_value_mixture)

                succ_trajs += (traj_states[-1][:default_state_len] == env.goal)*1
                if meta.found_obstacle and traj_states[-1][:default_state_len] == env.goal and len(meta.obstacle_hits) >= 1:
                    break

            # print('succ_trajs', succ_trajs)
            env.print_grid(trajectory=traj_states)
            print('Trajectory')
            print(traj_states)
            print('Obstacles hit')
            print(meta.obstacle_hits)
            print()
            print("Final belief:", meta.xi())
            print()
            print("Last step:", traj[-1] if traj else None)
            print()
