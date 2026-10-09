import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Set, Any
from collections import deque
import time
import numpy as np
import itertools

def generate_maze(width, height, scale=1, with_borders = False, just_obstacles=False, seed=123):

    random.seed(seed)
    
    # Ensure dimensions are odd
    w0 = width
    h0 = height
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    # Initialize the maze with walls
    maze = [[0 for _ in range(width)] for _ in range(height)] # 1 = Wall, 0 = Passage
    # maze = [[1 if x <= w0-(w0 % 2 and not with_borders)*1 and y <= h0-(h0 % 2 and not with_borders)*1 else 0 for x in range(width)] for y in range(height)] # 1 = Wall, 0 = Passage
    for x in range(width):
        for y in range(height):
            wshift = ((w0 % 2) != 0 and not with_borders and x == w0-1)*1
            hshift = ((h0 % 2) != 0 and not with_borders and y == h0-1)*1
            if wshift == 1:
                www = 1
            if hshift == 1:
                hhh = 1
            if x < width-wshift and y < height-hshift:
                maze[y][x] = 1  
            else:
                maze[y][x] = 0
                     
    bound_0 = 0 # 1 # with_borders*1
    def carve_passage(x, y):
        maze[y][x] = 0  # Carve the current cell
        
        # Define the possible directions (shuffled for randomness)
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)] # North, South, West, East (moving 2 steps to check unvisited cells)
        random.shuffle(directions)

        for dx, dy in directions:
            next_x, next_y = x + dx, y + dy

            # Check if the next cell is within bounds and is a wall (unvisited)
            if bound_0 <= next_x < width and bound_0 <= next_y < height and maze[next_y][next_x] == 1:
                # Carve the passage (wall between current and next cell)
                maze[y + dy // 2][x + dx // 2] = 0
                # Recursively call for the next cell

                if next_x < width and next_y < height:
                    carve_passage(next_x, next_y)
                      
    # Start carving from an initial cell (ensure it's an odd-numbered coordinate)
    carve_passage(1, 1)

    if just_obstacles:
        maze0 = [[maze[y][x] for x in range(width)] for y in range(height)]
        
        last_x = 0
        last_y = 0
        for y in range(1, height-1):
            for x in range(1, width-1):
                if maze0[last_y][last_x] == maze0[y][x]:
                    maze[y][x] = 0
                last_x = x
                last_y = y
    
        last_x = 0
        last_y = 0
        for x in range(1, width-1):
            for y in range(1, height-1):
                if maze0[last_y][last_x] == maze0[y][x]:
                    maze[y][x] = 0
                last_x = x
                last_y = y

    if not with_borders:
        if w0%2 == 0: width-=1
        if h0%2 == 0: height-=1
                                  
        maze = [[1 if i == 0 and maze[y][x+1] == 1 else 
                 1 if j == 0 and maze[y+1][x] == 1 else 
                 1 if i == width-1  and maze[y][x-1] == 1 and w0%2 != 0 else 
                 1 if j == height-1 and maze[y-1][x] == 1 and h0%2 != 0 else                      
                 maze[y][x] if i > 0 and j > 0 else 
                 0 
                 for i, x in enumerate(range(width))] for j, y in enumerate(range(height))]
    
    if scale > 1:
        new_maze = [[0 for _ in range(width*scale)] for _ in range(height*scale)] # 1 = Wall, 0 = Passage
        for x in range(width):
            for y in range(height):
                for sx in range(scale):
                    for sy in range(scale):
                        new_maze[y*scale + sy][x*scale + sx] = maze[y][x]
    else: 
        new_maze = maze

       
    return maze, new_maze

def print_maze(maze):
    for i, row in enumerate(maze):
        print("".join(["██" if cell == 1 else "  " for cell in row])+str(i))
    print("".join([((" " if i < 10 else "")+str(i)) for i,c in enumerate(row)]))


def doSeed(seed=None, verbose=True):
    if seed is None:
        seed = int(time.time()*1000) % (2**32) 
    else:
        seed %= (2**32) 
    
    if verbose: print(seed)
    np.random.seed(seed)
    random.seed(seed)

    return seed
    
State = Tuple[int, int]               # (x,y)
StateKD = Tuple[int, int, int, int]        # (x,y,has_key,crossed_door)  has_key in {0,1}, crossed_door in {0,1}


# =========================================================
# 1) Core environments (drop-in compatible + small extensions)
# =========================================================

class GridWorldPlus:
    """
    Minimal extension of your GridWorld to support:
      - walls (block movement, no terminal)
      - cliffs (heavy penalty, optional terminal)
      - traps (penalty, not terminal; enterable)
      - slippery transitions (stochastic action)
      - optional wind (push up/down/left/right probabilistically)
      - (optional) reward bonuses for special cells

    State is (x,y).
    """

    def __init__(
        self,
        width: int,
        height: int,
        start: State,
        goal: State,
        *,
        walls: Optional[Set[State]] = None,
        cliffs: Optional[Set[State]] = None,
        islands: Optional[Set[State]] = None,
        traps: Optional[Dict[State, float]] = None,
        bonus: Optional[Dict[State, float]] = None,
        # reward scaling (same spirit as your env)
        C: float = 10.0,
        k_border: float = 1.0,
        lambda_cliff: float = 10.0,
        goal_reward: float = 100,
        key_reward: float = 10,
        door_reward: float = 50,
        wall_reward_like_cliff: bool = False,
        # how cliffs behave
        cliff_end: bool = True,
        # whether border hits terminate (your greedy_agent flag behavior)
        border_end: bool = False,
        # stochasticity
        slip: Optional[Dict[State, float]] = None, # with prob slip_prob, replace intended action
        slip_mode: str = "uniform",  # "uniform" or "left_right" (perpendicular slip)
        # wind: dict cell -> (dx,dy,prob) applied after move (classic windy grid variants)
        wind: Optional[Dict[State, Tuple[int, int, float]]] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal

        self.walls = set(walls) if walls else set()
        self.cliffs = set(cliffs) if cliffs else set()
        self.islands = set(list(itertools.chain.from_iterable(islands))) if islands else set()
        self.traps = dict(traps) if traps else {}
        self.bonus = dict(bonus) if bonus else {}
        self.bonus0 = dict(bonus) if bonus else {}
        self.wind = dict(wind) if wind else {}
        self.slip = dict(slip) if slip else {}

        self.cliff_end = cliff_end
        self.border_end = border_end
        self.slip_mode = slip_mode

        self.key_cell = tuple()
        self.door_cells = set()
        self.has_key = 0
        self.crossed_door = 0

        # actions: 0=up, 1=right, 2=down, 3=left  (0=↑,1=→,2=↓,3=←)
        self.action_deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        # self.action_arrows = {0:'↑',1:'→',2:'↓',3:'←'}
        self.action_arrows = {'up':'↑','right':'→','down':'↓','left':'←'}        
        # self.action_arrows = {'up':'^','right':'>','down':'v','left':'<'}        
        self.action_size = len(self.action_deltas)
        self.action_space = list(range(self.action_size))

        if seed is None: seed = 123
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Reward scaling (kept close to your original)
        L_scale = (width + height) if (width + height) > 0 else 1
        self.r_step = float(np.round(-C / L_scale, 2))
        self.r_border = float(np.round(k_border * self.r_step, 2))
        self.r_cliff = float(np.round(-lambda_cliff * C / L_scale, 2))
        self.r_goal = float(np.round(goal_reward * C / L_scale, 2)) # 0.0
        self.r_key = float(np.round(key_reward * C / L_scale, 2)) # 0.0   
        self.r_door = float(np.round(door_reward * C / L_scale, 2)) # 0.0   

        self.r_key0 = self.r_key
        self.r_door0 = self.r_door

        self.r_wall = self.r_cliff if wall_reward_like_cliff else self.r_step

        self.name = name or f"{width}x{height}"
        self.state = start

    def update_r_wall(self, wall_reward_like_cliff):
        self.r_wall = self.r_cliff if wall_reward_like_cliff else self.r_step

    # ----- basics -----
    def reset(self, random_start: bool = False, start_state: Optional[State] = None) -> State:
        
        self.bonus = dict(self.bonus0)
        self.r_key = self.r_key0
        self.has_key = 0
        self.r_door = self.r_door0
        self.crossed_door = 0

        if start_state is not None:
            self.state = start_state
            return self.state

        if not random_start:
            self.state = self.start
            return self.state

        # random non-terminal, non-wall start
        while True:
            x = self.rng.randint(0, self.width - 1)
            y = self.rng.randint(0, self.height - 1)
            s = (x, y)
            if s != self.goal and not self.forbidden(s):
                self.state = s
                return self.state

    def _maybe_slip(self, s: State, action: int) -> int:
        
        if s not in self.slip:
            return action
            
        p = self.slip[s]
        
        if p <= 0.0:
            return action
            
        if self.rng.random() >= p:
            return action

        if self.slip_mode == "uniform":
            return self.rng.choice(self.action_space)

        if self.slip_mode == "left_right":
            # slip to perpendicular directions w.r.t intended move
            # up/down -> slip left/right; left/right -> slip up/down
            if action in (0, 2):
                return self.rng.choice([1, 3])
            return self.rng.choice([0, 2])

        # default fallback
        return self.rng.choice(self.action_space)

    def _apply_wind(self, s: State) -> State:
        """Apply wind after movement, if defined at resulting cell."""
        if s not in self.wind:
            return s
        dx, dy, p = self.wind[s]
        if self.rng.random() < p:
            nx, ny = s[0] + dx, s[1] + dy
            # wind respects borders & walls (clips by staying)
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.walls:
                return (nx, ny)
        return s

    def forbidden(self, s: State):
        return (s[0], s[1]) in self.cliffs or s in self.walls
                  
    def step(self, action: int):
        """
        Deterministic transition: from 'self.state' take 'action', 
        return (next_state, reward, done).
        """
        return self.step_from(self.state, action)
    
    def step_from(self, state: State, action: int):
        """
        Transition:
          1) maybe slip action
          2) attempt move
          3) handle border/wall
          4) apply wind
          5) compute reward & done (goal, cliffs, traps, bonus)
        """

        x, y = state[0], state[1]
        state = (x, y)

        a = self._maybe_slip(state, int(action))
        dx, dy = self.action_deltas[a]
        nx, ny = x + dx, y + dy

        # border -> stay
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            reward = self.r_border
            done = self.border_end
            self.state = (x, y)
            return self.state, reward, done

        cand = (nx, ny)

        # wall -> stay (no terminal)
        if cand in self.walls:
            cand = (x, y)
            reward = self.r_wall
        else:
            reward = self.r_step

        # door blocks unless has_key
        if cand in self.door_cells:
            if self.has_key == 0:
                cand = (x, y)
                reward = self.r_wall
            else:
                reward = self.r_door if self.r_door != 0 else self.r_step # + self.r_wall
                self.r_door = 0
                self.crossed_door = 1

        # apply wind after basic motion (wind uses cell-specific rules)
        cand = self._apply_wind(cand)
        nx, ny = cand

        # pick up key
        if cand == self.key_cell:
            reward += self.r_key if self.r_key != 0 else self.r_step
            self.r_key = 0
            self.has_key = 1

        done = (cand == self.goal)

        # reward composition
        if done:
            reward = self.r_goal  

        # entering a cliff cell: heavy penalty, optional terminal
        # (you can interpret cliffs as “hazard tiles” rather than forbidden)
        if cand in self.cliffs and not done:
            reward = self.r_cliff
            done = bool(self.cliff_end)
            cand = (x, y)

        # trap penalty if present (non-terminal unless you want to encode terminal traps yourself)
        if cand in self.traps and not done:
            reward += float(self.traps[cand])

        # bonus reward if present
        if cand in self.bonus and not done:
            reward += float(self.bonus[cand])
            self.bonus[cand] = 0

        self.state = cand
        return self.state, reward, done

    def all_states(self) -> List[State]:
        return [(i, j) for i in range(self.width) for j in range(self.height) if (i, j) not in (self.walls | self.islands)]

    def all_actions(self) -> List[int]:
        return list(range(self.action_size))

    def state_columns(self):
        return ('x', 'y') 

    def print_grid(self, extra_marks: Optional[Dict[State, str]] = None, trajectory: List[State] = None, print_rewards = False):
        """
        Simple blueprint print:
          S, G, # walls, C cliffs, T traps, + bonus
        """

        grid_str = (self.name+"\r\n")
        nsteps = 0 if trajectory is None else len(trajectory)
        if nsteps > 0:
            trajectory = [ (s[0], s[1]) for s in trajectory ]

        extra_marks = extra_marks or {}
        for j in range(self.height):
            row = []
            rew = []
            for i in range(self.width):
                s = (i, j)
                if s == self.start:
                    row.append("S")
                    rew.append(0.0)
                elif s == self.goal:
                    row.append("G")
                    rew.append(self.r_goal)
                elif s in self.islands:
                    row.append("X")
                    rew.append(self.r_wall)
                elif s in self.walls:
                    row.append("#")
                    rew.append(self.r_wall)
                elif s in self.cliffs:
                    row.append("*")
                    rew.append(self.r_cliff)
                else:
                    in_trajectory = False
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
                        in_trajectory = True
                        if ni > i:
                            row.append(self.action_arrows['right'])        
                        elif ni < i:
                            row.append(self.action_arrows['left'])        
                        elif nj < j:
                            row.append(self.action_arrows['up'])        
                        elif nj > j:
                            row.append(self.action_arrows['down'])   
                        else:
                            in_trajectory = False

                    if not in_trajectory:                    
                        if s in self.wind:
                            if self.wind[s][1] < 0:
                                row.append("^")
                            elif self.wind[s][1] > 0:
                                row.append("v")
                            elif self.wind[s][0] < 0:
                                row.append("<")
                            elif self.wind[s][0] > 0:
                                row.append(">")
                            else:
                                row.append("@")
                            rew.append(self.r_step)
                        elif s in self.slip:
                            row.append("~")                    
                            rew.append(self.r_step)
                        elif s in self.traps:
                            row.append("=")
                            rew.append(self.traps[s]+self.r_step)
                        elif s in self.bonus:
                            row.append("+")
                            rew.append(self.bonus[s]+self.r_step)
                        elif s in extra_marks:
                            row.append(extra_marks[s])
                            rew.append((self.r_key0 if extra_marks[s] == 'K' else self.r_door0 if extra_marks[s] == 'D' else 0) + self.r_step)
                        else:
                            row.append("·")
                            rew.append(self.r_step)
            if print_rewards:
                grid_str += " ".join(row) +"\t"+ " "*10 +"\t"+ " ".join([f"{r:6.2f}" for r in rew])+"\r\n"
            else:
                grid_str += " ".join(row)+"\r\n"
        grid_str += "\r\n"

        print(grid_str)
        return grid_str

class GridWorldKeyDoor(GridWorldPlus):
    """
    Key-Door variant:
      - state = (x, y, has_key, crossed_door)
      - door cells behave like walls unless has_key=1
      - key cell toggles has_key -> 1 when visited
      - crossing the door toggles crossed_door -> 1 when done

    Minimal changes: we override reset/step_from/all_states
    """

    def __init__(
        self,
        width: int,
        height: int,
        start: State,
        goal: State,
        *,
        key_cell: State,
        door_cells: Set[State],
        walls: Optional[Set[State]] = None,
        cliffs: Optional[Set[State]] = None,
        islands: Optional[Set[State]] = None,
        traps: Optional[Dict[State, float]] = None,
        bonus: Optional[Dict[State, float]] = None,
        C: float = 10.0,
        k_border: float = 1.0,
        lambda_cliff: float = 10.0,
        cliff_end: bool = True,
        border_end: bool = False,
        key_reward: float = 10.0,
        door_reward: float = 50.0,
        slip: Optional[Dict[State, float]] = None,
        slip_mode: str = "uniform",
        wind: Optional[Dict[State, Tuple[int, int, float]]] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            width, height, start, goal,
            walls=walls, cliffs=cliffs, islands=islands, traps=traps, bonus=bonus,
            C=C, k_border=k_border, lambda_cliff=lambda_cliff,
            cliff_end=cliff_end, border_end=border_end,
            slip=slip, slip_mode=slip_mode,
            key_reward=key_reward, door_reward=door_reward,
            wind=wind, seed=seed, name=name or f"{width}x{height}-KeyDoor"
        )
        self.key_cell = key_cell
        self.door_cells = set(door_cells)
        self.state: StateKD = (start[0], start[1], 0)

    def reset(self, random_start: bool = False, start_state: Optional[State] = None) -> State:
        start = super().reset(random_start, start_state)
        self.state = (start[0], start[1], 0, 0)

        return self.state

    def state_columns(self):
        return ('x', 'y' ,'has_key', 'crossed_door')

    def step_from(self, state: StateKD, action: int):
        xy_state, reward, done = super().step_from(state, action)
        self.state = (xy_state[0], xy_state[1], self.has_key, self.crossed_door)

        return self.state, reward, done

    def all_states(self) -> List[StateKD]:
        # All non-wall positions × {0,1}
        base  = [(i, j) for i in range(self.width) for j in range(self.height) if (i, j) not in self.walls]
        base1 = [(i, j, k) for (i, j) in base for k in (0, 1)]
        return  [(i, j, k, l) for (i, j, k) in base1 for l in (0, 1)]

    def print_grid(self, trajectory = None, print_rewards = False):
        extra = {self.key_cell: "K"}
        for d in self.door_cells:
            extra[d] = "D"
        super().print_grid(extra_marks=extra, trajectory=trajectory, print_rewards=print_rewards)


# =========================================================
# 2) Benchmark Families (each generates an env layout)
# =========================================================

@dataclass
class SizeSpec:
    width: int
    height: int


class BenchmarkFamily:
    family_name: str = "base"
    def make(self, size: SizeSpec, seed: Optional[int] = None):
        raise NotImplementedError


class CorridorFamily(BenchmarkFamily):
    family_name = "Corridor"
    def __init__(self, C=10.0, k_border=1.0, lambda_cliff: float = 10.0):
        self.C = C
        self.k_border = k_border
        self.lambda_cliff = lambda_cliff

    def make(self, size: SizeSpec, start=(0, -1), goal=(-1, 0), seed: Optional[int] = None) -> GridWorldPlus:
        w, h = size.width, size.height
        start = (start[0] if start[0] >= 0 else w+start[0], 
                 start[1] if start[1] >= 0 else h+start[1])
        goal = (goal[0] if goal[0] >= 0 else w+goal[0], 
                goal[1] if goal[1] >= 0 else h+goal[1])

        grid = GridWorldPlus(
            w, h,
            start=start, # (0, h-1),
            goal=goal, # (w-1, h-1),
            walls=set(),
            cliffs=set(),
            C=self.C, 
            k_border=self.k_border,
            lambda_cliff=self.lambda_cliff, 
            cliff_end=True,
            seed=seed,
            name=f"{self.family_name}-{w}x{h}"
        )

        grid.optimal_rets = np.round(grid.r_goal+abs(start[0]-goal[0]) + abs(start[1]-goal[1])*grid.r_step, 2) # /2 due to mean optimal return, depending on start state

        return grid

class CliffFamily(BenchmarkFamily):
    family_name = "Cliff"
    def __init__(self, cliff_end: bool = True, cliff_pos: str = "bottom", C=10.0, lambda_cliff=10.0):
        self.cliff_end = cliff_end
        self.cliff_pos = cliff_pos
        self.C = C
        self.lambda_cliff = lambda_cliff

    def make(self, size: SizeSpec, start=(0, -1), goal=(-1, 0), seed: Optional[int] = None) -> GridWorldPlus:
        w, h = size.width, size.height
        start = (start[0] if start[0] >= 0 else w+start[0], 
                 start[1] if start[1] >= 0 else h+start[1])
        goal = (goal[0] if goal[0] >= 0 else w+goal[0], 
                goal[1] if goal[1] >= 0 else h+goal[1])

        # classic cliff between start and goal along bottom row
        w_m = w // 3 # 10 
        h_m = w_m
        if self.cliff_pos == "bottom":
            cliffs ={(int(x), int(y)) for x in range(int(w_m/2),w-int(w_m/2)) for y in range(int(h-h_m), h)}
        else:
            cliffs ={(int(x), int(y)) for x in range(int(w_m/2),w-int(w_m/2)) for y in range(0, h_m)}
            
        grid = GridWorldPlus(
            w, h, start=start, goal=goal,
            cliffs=cliffs,
            C=self.C, lambda_cliff=self.lambda_cliff,
            cliff_end=self.cliff_end,
            seed=seed,
            name=f"{self.family_name}{f'-{self.cliff_pos[0].upper()+self.cliff_pos[1:].lower()}'}{'-Term' if self.cliff_end else '-Cont'}-{w}x{h}"
        )

        grid.optimal_rets = np.round(grid.r_goal+(w+h)*grid.r_step, 2) # /2 due to mean optimal return, depending on start state

        return grid


class SerpentMazeFamily(BenchmarkFamily):
    family_name = "Serpent"
    def __init__(self, corridor_width: int = 1, C=10.0, lambda_cliff: float = 10.0):
        """
        Produces a deterministic "serpentine corridor" maze:
          - alternating walls create a snake path.
        corridor_width controls how thick the corridor is (>=1).
        """
        self.corridor_width = max(1, int(corridor_width))
        self.C = C
        self.lambda_cliff = lambda_cliff

    def make(self, size: SizeSpec, start=(0, -1), goal=(-1, 0), seed: Optional[int] = None) -> GridWorldPlus:
        w, h = size.width, size.height

        start = (start[0] if start[0] >= 0 else w+start[0], 
                 start[1] if start[1] >= 0 else h+start[1])
        goal = (goal[0] if goal[0] >= 0 else w+goal[0], 
                goal[1] if goal[1] >= 0 else h+goal[1])

        walls = set()

        # Build serpentine barriers every few rows to force long detours.
        # For row blocks: create a wall across except a single gap alternating sides.
        scale = max(1, int(w // 10))
        self.corridor_width *= scale
        step = max(2*scale, self.corridor_width + 1)
        for y in range(scale, h-1, step):
            gap_on_left = ((y // step) % (2*scale) == 0)
            for x in range(w):
                # leave a corridor gap of corridor_width
                if gap_on_left:
                    if x < self.corridor_width:
                        continue
                else:
                    if x >= w - self.corridor_width:
                        continue
                for s in range(scale):
                    walls.add((x, y+s))       
            
        # Ensure start/goal not walls
        walls.discard(start); walls.discard(goal)

        grid = GridWorldPlus(
            w, h, start=start, goal=goal,
            walls=walls,
            C=self.C,
            lambda_cliff = self.lambda_cliff,
            seed=seed,
            name=f"{self.family_name}-Maze-{w}x{h}"
        )

        grid.optimal_rets = np.round(grid.r_goal+(w*(h//step+1)+h)*grid.r_step, 2) # /2 due to mean optimal return, depending on start state

        return grid

Cell = Tuple[int, int]

def find_closed_rooms(width: int, height: int, obstacles: Set[Cell]) -> Set[Cell]:
    """
    Return all free cells that belong to closed rooms in a grid.

    A closed room is any connected component of non-obstacle cells that does
    NOT touch the outer border of the grid.

    Assumptions:
    - Coordinates are (x, y)
    - 0 <= x < width and 0 <= y < height
    - Connectivity is 4-directional: up, down, left, right

    Parameters
    ----------
    width : int
        Grid width.
    height : int
        Grid height.
    obstacles : set[tuple[int, int]]
        Set of blocked cells.

    Returns
    -------
    set[tuple[int, int]]
        All cells that are inside closed rooms. Empty set if none exist.
    """
    if width <= 0 or height <= 0:
        return set()

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    def neighbors(x: int, y: int):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny):
                yield nx, ny

    visited: Set[Cell] = set()
    closed_cells: Set[Cell] = set()

    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles or (x, y) in visited:
                continue

            # Explore one connected component of free cells
            queue = deque([(x, y)])
            component = set()
            touches_border = False
            visited.add((x, y))

            while queue:
                cx, cy = queue.popleft()
                component.add((cx, cy))

                if cx == 0 or cx == width - 1 or cy == 0 or cy == height - 1:
                    touches_border = True

                for nx, ny in neighbors(cx, cy):
                    if (nx, ny) not in obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

            if not touches_border:
                closed_cells.update(component)

    return closed_cells


def find_unreachable_clusters(
    width: int,
    height: int,
    obstacles: Set[Cell],
    goal: Cell
) -> List[List[Cell]]:
    """
    Find clusters of free cells that cannot reach the goal.

    Parameters
    ----------
    width : int
        Grid width.
    height : int
        Grid height.
    obstacles : set[(int, int)]
        Blocked cells.
    goal : (int, int)
        Goal cell.

    Returns
    -------
    list[list[(int, int)]]
        A list of unreachable clusters. Each cluster is a list of cells.
    """
    if width <= 0 or height <= 0:
        return []

    gx, gy = goal

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    def neighbors(x: int, y: int):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny):
                yield nx, ny

    # If goal is invalid or blocked, then every free-cell component is unreachable.
    goal_is_valid_free = in_bounds(gx, gy) and goal not in obstacles

    reachable = set()

    if goal_is_valid_free:
        queue = deque([goal])
        reachable.add(goal)

        while queue:
            x, y = queue.popleft()
            for nx, ny in neighbors(x, y):
                cell = (nx, ny)
                if cell not in obstacles and cell not in reachable:
                    reachable.add(cell)
                    queue.append(cell)

    visited = set()
    clusters = []

    for y in range(height):
        for x in range(width):
            cell = (x, y)

            if cell in obstacles or cell in reachable or cell in visited:
                continue

            # New unreachable component
            cluster = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                cx, cy = queue.popleft()
                cluster.append((cx, cy))

                for nx, ny in neighbors(cx, cy):
                    ncell = (nx, ny)
                    if (
                        ncell not in obstacles
                        and ncell not in reachable
                        and ncell not in visited
                    ):
                        visited.add(ncell)
                        queue.append(ncell)

            clusters.append(cluster)

    return clusters

class RandomMazeFamily(BenchmarkFamily):
    family_name = "Random"
    def __init__(self, corridor_width: int = 1, C=10.0, lambda_cliff: float = 10.0):
        """
        Produces a deterministic "serpentine corridor" maze:
          - alternating walls create a snake path.
        corridor_width controls how thick the corridor is (>=1).
        """
        self.C = C
        self.lambda_cliff = lambda_cliff

    def make(self, size: SizeSpec, start=(0, -1), goal=(-1, 0), scale = -1, walls_perc = 0, 
             seed: int = None, obstacle_seed: int = None, verbose = True) -> GridWorldPlus:
        w, h = size.width, size.height
        
        start = (start[0] if start[0] >= 0 else w+start[0], 
                 start[1] if start[1] >= 0 else h+start[1])
        goal = (goal[0] if goal[0] >= 0 else w+goal[0], 
                goal[1] if goal[1] >= 0 else h+goal[1])

        walls = set()

        if seed is None or seed==123:
            seed = 1928578606
        seed=doSeed(seed, verbose=verbose)

        scale = max(1, int(w // 20)) if scale <= 0 else scale
        
        mazep, maze = generate_maze(w // scale, h // scale, scale, with_borders=False, just_obstacles=False, seed=seed)

        _ = [[walls.add((x, y)) if maze[y][x] else None for x in range(w)] for y in range(h)]
                    
        # Ensure start/goal not walls
        walls.discard(start); walls.discard(goal)

        if 0 < walls_perc < 1:
            num_all_walls = len(walls)
            num_final_walls = int(num_all_walls * walls_perc)

            if obstacle_seed is None:
                final_walls_pos = sorted(np.random.choice(num_all_walls, num_final_walls, replace=False))
            else:
                # rng_obs = np.random.default_rng(obstacle_seed if obstacle_seed is not None else seed)
                # final_walls_pos = sorted(rng_obs.choice(num_all_walls, num_final_walls, replace=False))        
                np.random.seed(obstacle_seed if obstacle_seed is not None else seed)        
                final_walls_pos = sorted(np.random.choice(num_all_walls, num_final_walls, replace=False))

            sorted_walls = np.array(sorted(list(walls)))
            new_walls = sorted_walls[final_walls_pos]
            walls = set([tuple(w) for w in new_walls])

            # closed_rooms = [(x, y) for x in range(w) if 0 < x < w for y in range(h) if 0 < y < h and (x, y-1) in walls and (x, y+1) in walls and (x-1, y) in walls and (x+1, y) in walls]
            # closed_rooms = find_closed_rooms(w, h, walls)
            # if closed_rooms:
            #     walls.update(closed_rooms)

            walls_perc_str = f'-{walls_perc}'
        else:
            walls_perc_str = ''
            
        # walls.update([(6, 7)]) # test to force an island

        islands = find_unreachable_clusters(w, h, walls, goal)
        if islands and verbose:
            for i, island in enumerate(islands[:], start=1):
                print(f"Island {i}: size={len(island)}, sample={island[:]}")

        grid =  GridWorldPlus(
            w, h, start=start, goal=goal, islands=islands,
            walls=walls,
            C=self.C,
            lambda_cliff = self.lambda_cliff,
            seed=seed,
            name=f"{self.family_name}-Maze{walls_perc_str}-{w}x{h}"
        )
        
        grid.optimal_rets = np.round(grid.r_goal+(w+h-2-1)*1.0*grid.r_step, 2) # /2 due to mean optimal return, depending on start state

        return grid

# Add these two helpers near find_unreachable_clusters in the same file.

def _is_connected(w: int, h: int, walls: Set[Cell], start: Cell, goal: Cell) -> bool:
    """BFS from start over non-wall cells; True iff goal is reachable."""
    if start in walls or goal in walls:
        return False
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < w and 0 <= ny < h
                    and (nx, ny) not in seen and (nx, ny) not in walls):
                seen.add((nx, ny))
                queue.append((nx, ny))
    return False


def _carve_minimum_walls(w: int, h: int, walls: Set[Cell],
                         start: Cell, goal: Cell) -> Set[Cell]:
    """
    Remove the minimum number of walls so that start can reach goal.
    Uses Dijkstra in the full 4-connected grid where the edge cost into
    a non-wall cell is 0 and into a wall cell is 1.  We then backtrack
    from goal to start and discard the walls along the optimal path.

    The returned set is `walls` minus the carved cells.  In the worst
    case the carve removes O(W+H) cells along a Manhattan path.
    """
    import heapq
    INF = float('inf')
    dist = {start: 0}
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    pq: List[Tuple[int, Cell]] = [(0, start)]

    while pq:
        d, (x, y) = heapq.heappop(pq)
        if (x, y) == goal:
            break
        if d > dist.get((x, y), INF):
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            cost = 1 if (nx, ny) in walls else 0
            nd = d + cost
            if nd < dist.get((nx, ny), INF):
                dist[(nx, ny)] = nd
                parent[(nx, ny)] = (x, y)
                heapq.heappush(pq, (nd, (nx, ny)))

    if goal not in dist:
        return walls  # genuinely unreachable; caller should resample

    out = set(walls)
    cur: Optional[Cell] = goal
    while cur is not None:
        out.discard(cur)
        cur = parent[cur]
    return out

class RandomScatterFamily(BenchmarkFamily):
    """
    Uniform random scatter of obstacles.  Unlike RandomMazeFamily, this
    family does NOT carve a DFS-style maze skeleton -- obstacles are
    placed uniformly at random over all grid cells (excluding start and
    goal).

    Why this exists
    ---------------
    The percolation prediction
        Pr[c in F*]  =  (1 - rho)^N
    assumes obstacles are independently sampled per cell over the full
    grid.  RandomMazeFamily violates this in two ways: (i) walls_perc
    is the fraction of *skeleton walls* kept, not the fraction of grid
    cells, and (ii) DFS-carved mazes have substantial structural
    overlap (border patterns, alternating-cell wall constraints) that
    make different seeds correlated even before sub-sampling.  Both
    effects inflate F*-density above the i.i.d. prediction.

    RandomScatterFamily eliminates both: walls_perc IS the per-cell
    obstacle probability, and two seeds produce truly independent
    obstacle patterns (modulo the connectivity-enforcement step).

    The cost is that the result no longer looks like a "maze" in the
    visual sense -- it is a random obstacle scatter that happens to
    admit a start->goal path.  For the percolation experiment, this is
    a feature: the theoretical prediction was always derived for the
    scatter regime.

    Parameters
    ----------
    connectivity_mode : {'carve', 'retry', 'none'}, default 'carve'
        How to ensure each individual env is solvable (s0 -> g path
        exists):
          'carve': sample obstacles, then if disconnected remove the
              minimum number of obstacles along a shortest 0-1 BFS
              path.  Cheap and works at any rho; slightly biases
              effective density downward by O((W+H)*P_disconnected)
              cells per env.
          'retry': sample obstacles; if disconnected, advance the seed
              and resample; up to max_retries attempts before falling
              back to 'carve'.  Cleanest statistics for rho < ~0.5;
              loops near percolation threshold.
          'none' : do not enforce connectivity.  Only use if you want
              to study the disconnected regime directly.
    max_retries : int, default 50
        Cap on resampling attempts in 'retry' mode.
    """
    family_name = "Scatter"

    def __init__(self,
                 C: float = 10.0,
                 lambda_cliff: float = 10.0,
                 connectivity_mode: str = 'carve',
                 max_retries: int = 50):
        if connectivity_mode not in ('carve', 'retry', 'none'):
            raise ValueError(f"connectivity_mode must be 'carve', 'retry', "
                             f"or 'none'; got {connectivity_mode!r}")
        self.C = C
        self.lambda_cliff = lambda_cliff
        self.connectivity_mode = connectivity_mode
        self.max_retries = max_retries

    def make(self,
             size: SizeSpec,
             start: Tuple[int, int] = (0, -1),
             goal: Tuple[int, int] = (-1, 0),
             scale: int = -1,           # accepted for API parity; ignored
             walls_perc: float = 0.0,
             seed: Optional[int] = None,
             obstacle_seed: int = None, 
             verbose: bool = True) -> 'GridWorldPlus':
        w, h = size.width, size.height

        # Resolve negative indexing in start/goal (matches RandomMazeFamily).
        start = (start[0] if start[0] >= 0 else w + start[0],
                 start[1] if start[1] >= 0 else h + start[1])
        goal  = (goal[0]  if goal[0]  >= 0 else w + goal[0],
                 goal[1]  if goal[1]  >= 0 else h + goal[1])

        # Seed handling matches the maze family's defaults.
        if seed is None or seed == 123:
            seed = 1928578606
        seed = doSeed(seed, verbose=verbose)

        # Sample walls.  Note walls_perc is now over the FULL grid.
        walls = self._sample_walls(w, h, start, goal, walls_perc, verbose)

        islands = find_unreachable_clusters(w, h, walls, goal)
        if islands and verbose:
            for i, island in enumerate(islands, start=1):
                print(f"Island {i}: size={len(island)}, sample={island[:]}")

        walls_perc_str = f'-{walls_perc}' if walls_perc > 0 else ''
        grid = GridWorldPlus(
            w, h, start=start, goal=goal, islands=islands,
            walls=walls,
            C=self.C,
            lambda_cliff=self.lambda_cliff,
            seed=seed,
            name=f"{self.family_name}-Maze{walls_perc_str}-{w}x{h}",
        )
        grid.optimal_rets = np.round(
            grid.r_goal + (w + h - 2 - 1) * 1.0 * grid.r_step, 2,
        )
        return grid

    # ------------------------------------------------------------
    # internal: obstacle sampling with optional connectivity guard
    # ------------------------------------------------------------
    def _sample_walls(self, w: int, h: int,
                      start: Cell, goal: Cell,
                      walls_perc: float,
                      verbose: bool) -> Set[Cell]:
        """
        Returns a set of obstacle cells.  walls_perc is the per-cell
        obstacle probability over the full grid (excluding start and
        goal).  The exact obstacle count is round(walls_perc * w * h),
        clamped to leave start and goal free.
        """
        # Candidate cells: everything except start and goal.
        candidates: List[Cell] = [
            (x, y) for y in range(h) for x in range(w)
            if (x, y) != start and (x, y) != goal
        ]
        n_cells = w * h
        n_walls = int(round(walls_perc * n_cells))
        n_walls = max(0, min(n_walls, len(candidates)))
        if n_walls == 0:
            return set()

        cand_arr = np.array(candidates, dtype=np.int64)

        def draw_subset() -> Set[Cell]:
            idx = np.random.choice(len(candidates), n_walls, replace=False)
            return set(map(tuple, cand_arr[idx].tolist()))

        # 'none' mode: no guard.
        if self.connectivity_mode == 'none':
            return draw_subset()

        # 'retry' mode: resample until connected, then fall through.
        if self.connectivity_mode == 'retry':
            for attempt in range(self.max_retries):
                walls = draw_subset()
                if _is_connected(w, h, walls, start, goal):
                    return walls
            if verbose:
                print(f"[Random-Scatter] {self.max_retries} retries "
                      f"exhausted at walls_perc={walls_perc}; "
                      f"falling back to carve.")

        # 'carve' mode (or fallback from exhausted retry).
        walls = draw_subset()
        if not _is_connected(w, h, walls, start, goal):
            walls = _carve_minimum_walls(w, h, walls, start, goal)
        return walls
    
class TrapBonusFamily(BenchmarkFamily):
    family_name = "Trap-Bonus"
    def __init__(self, trap_bonus_reward: float = 0.0, trap_size: Tuple[int, int] = (3, 2), C=10.0):
        """
        trap_bonus_reward is added to r_step when inside trap cells (negative means worse).
        trap_size = (tw, th) rectangle region.
        """
        self.trap_bonus_reward = float(trap_bonus_reward)
        self.trap_size = trap_size
        self.C = C

    def make(self, size: SizeSpec, start=(0, -1), goal=(-1, 0), seed: Optional[int] = None) -> GridWorldPlus:
        w, h = size.width, size.height        
        start = (start[0] if start[0] >= 0 else w+start[0], 
                 start[1] if start[1] >= 0 else h+start[1])
        goal = (goal[0] if goal[0] >= 0 else w+goal[0], 
                goal[1] if goal[1] >= 0 else h+goal[1])
       
        scale = max(1, (w+9) // 10)
        # Put a trap region near the "direct" diagonal-ish path
        tw, th = (c*scale for c in self.trap_size)

        trap_bonus_reward0 = self.trap_bonus_reward
        L_scale = (w + h) 
        self.trap_bonus_reward = np.round(float(self.trap_bonus_reward) * self.C / L_scale / (tw * th) / 2, 2)
        
        x0 = max(1, (w // 2) - (tw // 2))
        y0 = max(1, (h // 2) - (th // 2))
        rewards = {}
        for yy in range(y0, min(h-1, y0 + th)):
            for xx in range(x0, min(w-1, x0 + tw)):
                rewards[(xx, yy)] = self.trap_bonus_reward

        grid = GridWorldPlus(
            w, h, start=start, goal=goal,
            traps=rewards if self.trap_bonus_reward < 0 else None,
            bonus=rewards if self.trap_bonus_reward > 0 else None,
            C=self.C,
            seed=seed,
            name=f"{self.family_name}{('+'if trap_bonus_reward0 > 0 else '')+str(trap_bonus_reward0)}-{w}x{h}"
        )

        grid.optimal_rets = np.round(grid.r_goal+(w+h)*grid.r_step, 2) # /2 due to mean optimal return, depending on start state

        return grid


class SlipperyFamily(BenchmarkFamily):
    family_name = "Slippery"
    def __init__(self, slip_prob: float = 0.2, slippery_band_height: float = 0.4, C=10.0):
        """
        Slippery transitions everywhere OR in a band (implemented as global slip_prob here).
        If you want region-specific slip, we can add a slip_mask easily.
        """
        self.slip_prob = float(slip_prob)
        self.slippery_band_height = float(slippery_band_height)
        self.C = C

    def make(self, size: SizeSpec, start=(0, -1), goal=(-1, 0), seed: Optional[int] = None) -> GridWorldPlus:
        w, h = size.width, size.height
        start = (start[0] if start[0] >= 0 else w+start[0], 
                 start[1] if start[1] >= 0 else h+start[1])
        goal = (goal[0] if goal[0] >= 0 else w+goal[0], 
                goal[1] if goal[1] >= 0 else h+goal[1])

        # Put wind in middle columns pushing up
        slip = {}
        for x in range(w // 3, 2 * w // 3):
            for y in range(int(h*self.slippery_band_height/2), h-int(h*self.slippery_band_height/2)):
                slip[(x, y)] = self.slip_prob  # push up
        
        # simplest: global slip
        grid = GridWorldPlus(
            w, h, start=start, goal=goal,
            slip=slip,
            slip_mode="uniform",
            C=self.C,
            seed=seed,
            name=f"{self.family_name}-p{int(self.slip_prob*100)}-{w}x{h}"
        )

        grid.optimal_rets = np.round(grid.r_goal+(w+h)*grid.r_step, 2) # /2 due to mean optimal return, depending on start state

        return grid

class WindyFamily(BenchmarkFamily):
    family_name = "Windy"
    def __init__(self, wind_prob: float = 0.5, C=10.0):
        self.wind_prob = float(wind_prob)
        self.C = C

    def make(self, size: SizeSpec, start=(0, -1), goal=(-1, 0), wind_dir = (0, -1), seed: Optional[int] = None) -> GridWorldPlus:
        w, h = size.width, size.height
        start = (start[0] if start[0] >= 0 else w+start[0], 
                 start[1] if start[1] >= 0 else h+start[1])
        goal = (goal[0] if goal[0] >= 0 else w+goal[0], 
                goal[1] if goal[1] >= 0 else h+goal[1])

        # Put wind in middle columns pushing up
        wind = {}
        for x in range(w // 3, 2 * w // 3):
            for y in range(h):
                wind[(x, y)] = (wind_dir[0], wind_dir[1], self.wind_prob)  # push up

        grid =  GridWorldPlus(
            w, h, start=start, goal=goal,
            wind=wind,
            C=self.C,
            seed=seed,
            name=f"{self.family_name}-p{int(self.wind_prob*100)}-{w}x{h}"
        )

        grid.optimal_rets = np.round(grid.r_goal+(w+h)*grid.r_step, 2) # /2 due to mean optimal return, depending on start state

        return grid

class KeyDoorFamily(BenchmarkFamily):
    family_name = "KeyDoor"
    def __init__(self, C=10.0, key_reward: float = 10, door_reward: float = 50, lambda_cliff: float = 10.0):
        self.C = C
        self.lambda_cliff = lambda_cliff
        self.key_reward = key_reward
        self.door_reward = door_reward

    def make(self, size: SizeSpec, start=(0, -1), goal=(-1, 0), seed: Optional[int] = None) -> GridWorldKeyDoor:
        w, h = size.width, size.height
        start = (start[0] if start[0] >= 0 else w+start[0], 
                 start[1] if start[1] >= 0 else h+start[1])
        goal = (goal[0] if goal[0] >= 0 else w+goal[0], 
                goal[1] if goal[1] >= 0 else h+goal[1])

        # Build a horizontal wall barrier with one door
        walls = set()
        barrier_y = h // 2 - 1
        for x in range(w):
            walls.add((x, barrier_y))

        # Door location somewhere on barrier
        door = (w - 2, barrier_y)
        door_cells = {door}
        walls.discard(door)

        # Key placed on start side (below barrier if start is bottom)
        # key = (w // 3, h - 2)
        key = (w // 3, barrier_y + 2)

        grid = GridWorldKeyDoor(
            w, h, start=start, goal=goal,
            walls=walls,
            key_cell=key,
            key_reward=self.key_reward,
            door_reward=self.door_reward,
            door_cells=door_cells,
            C=self.C,
            lambda_cliff = self.lambda_cliff,
            seed=seed,
            name=f"{self.family_name}-{w}x{h}"
        )

        grid.optimal_rets = np.round(grid.r_goal+
                                     grid.r_key+
                                     grid.r_door+
                                     (w+h)*grid.r_step, 2) # /2 due to mean optimal return, depending on start state

        return grid

# =========================================================
# 3) Suite builder
# =========================================================

def build_canonical_suite(
    sizes: Sequence[Tuple[int, int]],
    random_maze_only = False,
    lambda_cliff=100.0,
    seed: int = None,
    num_percs = 10,
    perc0 = 0.0,
    percN = 1.0,
    nested_grids = True, 
    verbose=True,    
) -> Dict[Tuple[str, str], Any]:
    """
    Returns dict[(family_name, env_name)] -> env_instance
    """
    
    key_reward = 20
    door_reward = 50
    
    if nested_grids:
        seeds = [None]*num_percs
    else:

        seedN = 1928578606 if seed is None or seed == 123 else seed # original seed to always generate the same env with 1.0 obstacle concentration
        seeds = [0]*num_percs
        step = max(1, 40 // num_percs)
        seed = seedN // 2
        for i in range(0, max(40, num_percs), step):
            for s in range(step):
                seed=doSeed(seed, verbose=verbose)        
                seed += random.randint(0, seed)
                seeds[i//step] = seed % (2**32) 
        seeds[-1] = seedN

    if verbose: print(seeds)

    if random_maze_only:
        families: List[BenchmarkFamily] = [ (RandomMazeFamily(C=10.0, lambda_cliff=lambda_cliff), 10000) for (w, h) in sizes ]

        suite = {}
        for s, (w, h) in enumerate(sizes):
            size = SizeSpec(w, h)
            (fam, eps) = families[s]
            for i in range(num_percs):
                perc = ((i+1) / num_percs)
                seed = seeds[i]
                if perc0 <= perc <= percN:
                    if verbose: print( i, eps, fam.family_name )
                    env = fam.make(size=size, scale=1, walls_perc=perc, seed=seed, verbose=verbose)
                    suite[(fam.family_name, env.name, i)] = (env, eps)     
     
    else:
        families: List[BenchmarkFamily] = [
            (CorridorFamily(C=10.0, lambda_cliff=lambda_cliff), 500),
            (CliffFamily(cliff_end=False, cliff_pos="bottom", C=10.0, lambda_cliff=lambda_cliff), 500),
            (CliffFamily(cliff_end=False, cliff_pos="top", C=10.0, lambda_cliff=lambda_cliff), 500),
            (SerpentMazeFamily(C=10.0, lambda_cliff=lambda_cliff), 5000),
            (RandomMazeFamily(C=10.0, lambda_cliff=lambda_cliff), 10000),            
            (RandomScatterFamily(C=10.0, lambda_cliff=lambda_cliff), 10000),
            (TrapBonusFamily(trap_bonus_reward=-10.0, trap_size=(3, 2), C=10.0), 1000),
            (TrapBonusFamily(trap_bonus_reward=100.0, trap_size=(3, 2), C=10.0), 5000),
            (SlipperyFamily(slip_prob=0.2, C=10.0), 1000),
            (WindyFamily(wind_prob=0.5, C=10.0), 1000),
            (KeyDoorFamily(C=10.0, lambda_cliff=lambda_cliff, key_reward=key_reward, door_reward=door_reward), 10000),
        ]

        suite = {}
        for (w, h) in sizes:
            size = SizeSpec(w, h)
            for i, (fam, eps) in enumerate(families):
                if verbose: print( i, eps, fam.family_name )
                env = fam.make(size=size, seed=seed)
                suite[(fam.family_name, env.name, i)] = (env, eps)

    return suite


# =========================================================
# 4) Quick demo
# =========================================================
if __name__ == "__main__":
    sizes = [(10, 5), (20, 10), (40, 20), (60,30), (80, 40), (100, 50)]
    # sizes = [(10,5)] 
    # sizes = [(20,10)] 
    sizes = [(40,20)]
    # sizes = [(40,25)]  
    # sizes = [(60,30)] 
    # sizes = [(80,40)] 
    # sizes = [(100,50)] 
    # lambda_cliff=100.0
    # fam = RandomMazeFamily(C=10.0, lambda_cliff=lambda_cliff)
    # env = fam.make(size=SizeSpec(40,25), scale=1, walls_perc=0.2, seed=1928578606)

    num_percs = 40
    perc0 = 0.5
    percN = 0.75        

    num_percs = 10 # 20 # 
    perc0 = 0
    percN = 1.0        

    nested_grids = True # False # 

    suite = build_canonical_suite(sizes, random_maze_only=True, seed=123,
                                 num_percs=num_percs, nested_grids=nested_grids, perc0=perc0, percN=percN)

    print_rewards = False
    
    # Print one example per family at smallest size
    printed = set()
    for (fam, env_name, i), env in suite.items():
        if True or (env_name.endswith("10x4") and fam not in printed):
            print(f"=== {fam} / {env_name} ===")
            env[0].print_grid(print_rewards=print_rewards)
            printed.add(fam)

    
    envs_eps = [ (env, n_episodes) for (fam, env_name, i), (env, n_episodes) in suite.items() ]

    envs, num_episodes_list = zip(*envs_eps)    
    N = len(envs)//len(sizes)

    for i, env in enumerate(envs):
        print(env.name)

    print(f'N={N}')
    print()
