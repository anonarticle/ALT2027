
## Non-Torch Self-Supervised Bayesian MetaAgent (no reward thresholds)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional, Protocol
import math
import random
import numpy as np
from collections import defaultdict

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

value_mixture_action_temperature = 0.3

# ------------ Environment protocol ------------

class Environment(Protocol):
    """
    Minimal interface expected by MetaAgent.
    You can adapt your own Environment class to this.
    """
    width: int
    height: int
    start_state: State
    max_steps: int

    def reset(self) -> State: ...
    # def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]: ...
    def step(self, action: Action) -> Tuple[State, float, bool]: ...
    def step_from(self, s: State, action: Action) -> Tuple[State, float, bool]: ...

# ------------ helpers ------------

def normalize_probs(p: List[float], eps: float = 1e-12) -> List[float]:
    s = float(sum(p))
    if s < eps:
        # fallback to uniform if degenerate
        return [1.0 / len(p)] * len(p)
    return [float(x) / s for x in p]

def logsumexp(xs: List[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs) + 1e-300)

def logsumexp2(a: float, b: float) -> float:
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m) + 1e-300)

def softmax(logits: List[float], temperature: float =1.0) -> List[float]:
    logits = np.array(logits) / max(temperature, 1e-6) 
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def softmax_row(z, temperature=1.0):
    z = z / max(temperature, 1e-6) # / np.sum(z)
    m = np.max(z)
    e = np.exp(np.clip(z - m, -50, 50))
    return e / (np.sum(e) + 1e-12)

def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v

def sigmoid_stable(x: float) -> float:
    # stable for very large |x|
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)
    
@dataclass
class StepRecord:
    episode: int
    t: int 
    s: State
    a: Action
    # s_next: State
    r: float
    #done: bool
    # belief: List[float]          # posterior after update

# ---------------- Reward model: Normal-Inverse-Gamma -> Student-t predictive ----------------

@dataclass
class NormalInverseGamma:
    """
    Normal-Inverse-Gamma prior/posterior for unknown (mu, sigma^2).

    Prior:
      mu | sigma^2 ~ Normal(mu0, sigma^2/kappa0)
      sigma^2 ~ Inv-Gamma(alpha0, beta0)

    With weighted observations, we update sufficient stats using "fractional counts".
    Predictive for a new r is Student-t.
    """
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 2.0
    beta0: float = 2.0

    # posterior params (start at prior)
    mu: float = 0.0
    kappa: float = 1.0
    alpha: float = 2.0
    beta: float = 2.0

    def __post_init__(self):
        self.mu = float(self.mu0)
        self.kappa = float(self.kappa0)
        self.alpha = float(self.alpha0)
        self.beta = float(self.beta0)

    def update_weighted(self, x: float, w: float) -> None:
        """
        Weighted conjugate update with fractional weight w.
        Treat as w "pseudo-observations" at value x.
        """
        if w <= 0.0:
            return
        # Equivalent to sequentially updating with an observation weighted by w.
        # For NormalInverseGamma, we can do closed-form for a batch of size w at identical x:
        # kappa' = kappa + w
        # mu' = (kappa*mu + w*x) / (kappa+w)
        # alpha' = alpha + w/2
        # beta' = beta + 0.5 * (kappa*w/(kappa+w)) * (x - mu)^2
        kappa_new = self.kappa + w
        mu_new = (self.kappa * self.mu + w * x) / kappa_new
        alpha_new = self.alpha + 0.5 * w
        beta_new = self.beta + 0.5 * (self.kappa * w / kappa_new) * (x - self.mu) ** 2

        self.kappa, self.mu, self.alpha, self.beta = kappa_new, mu_new, alpha_new, beta_new

    def log_predictive(self, x: float) -> float:
        """
        log p(x | posterior) under Student-t predictive:
          df = 2*alpha
          loc = mu
          scale^2 = beta*(kappa+1)/(alpha*kappa)
        """
        df = 2.0 * self.alpha
        loc = self.mu
        scale2 = (self.beta * (self.kappa + 1.0)) / (self.alpha * self.kappa + 1e-12)
        scale = math.sqrt(scale2 + 1e-12)

        # Student-t logpdf
        z = (x - loc) / (scale + 1e-12)
        # log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(νπ) - log(scale) - ((ν+1)/2) log(1 + z^2/ν)
        nu = df
        return (
            math.lgamma((nu + 1.0) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * math.log(nu * math.pi + 1e-300)
            - math.log(scale + 1e-300)
            - ((nu + 1.0) / 2.0) * math.log(1.0 + (z * z) / (nu + 1e-300))
        )


# ---------------- Cell reward container: two NIGs ----------------

@dataclass
class TwoNIGCell:
    """
    Two NormalInverseGamma for a given target cell:
      mode0: empty-like
      mode1: obstacle-like
    """
    nig0: NormalInverseGamma
    nig1: NormalInverseGamma

    # diagnostics: effective masses
    total_mass: float = 0.0
    mass0: float = 0.0
    mass1: float = 0.0

# ---------------- Transition model: Dirichlet-Categorical over next-state ----------------

class DirichletNextState:
    """
    For each (s,a), maintain counts over s' with symmetric Dirichlet prior alpha.
    Supports stochastic transitions naturally.
      p(s'|s,a) = (c(s,a,s') + alpha) / (C(s,a) + alpha*K)
    """
    def __init__(self, num_states: int, alpha: float = 0.5):
        self.num_states = int(num_states)
        self.alpha = float(alpha)
        # counts[(s,a)] -> dict[sprime_idx] = count
        self.counts = defaultdict(lambda: defaultdict(float))
        self.totals = defaultdict(float)

    def update_weighted(self, sa_key: Tuple[int, int], sprime_idx: int, w: float) -> None:
        if w <= 0.0:
            return
        self.counts[sa_key][sprime_idx] += w
        self.totals[sa_key] += w

    def log_predictive(self, sa_key: Tuple[int, int], sprime_idx: int) -> float:
        # p = (c + alpha) / (total + alpha*K)
        c = self.counts[sa_key].get(sprime_idx, 0.0)
        total = self.totals.get(sa_key, 0.0)
        p = (c + self.alpha) / (total + self.alpha * self.num_states + 1e-300)
        return math.log(p + 1e-300)


# ---------------- MetaAgent: Dirichlet transitions + obstacle-gated mixture of 2 Student-t ----------------

class MetaAgent2NIGSelfSup:
    """
    Compact module that plugs into the Dirichlet transition + Bayes belief loop with minimal disruption.

    Observation: o_t=(s,a,r,s')
    Per-env likelihood:
      p_ν(o_t) = p_ν(s'|s,a) * p_ν(r|t)
    where t is the attempted target cell from (s,a).

    Reward likelihood uses obstacle-map gating (given):
      w_ν(t) = p_obs^ν(t) in [0,1]
      p_ν(r|t) = (1-w_ν(t)) * St0_t(r) + w_ν(t) * St1_t(r)

    Each target cell t has two NIG posteriors (shared across envs):
      Stk_t is Student-t predictive induced by NIG_k,t.
    Updates are self-supervised:
      - env posterior γ_env(ν) from Bayes
      - mode posterior P(z=1|ν,r,t)
      - aggregate w1_total = Σν γ_env(ν) P(z=1|ν,r,t)
      - update NIG1_t with weight w1_total, NIG0_t with weight 1-w1_total
    """

    def __init__(
        self,
        env_template: Environment,
        q_tables: List[Dict[State, List[float]]],
        action_space: List[Any],
        epsilon: float = 0.05,
        # priors / smoothing
        trans_alpha: float = 0.5,
        obs_sharpness: float = 6.0,        
        use_belief_weighted_Q: bool = True,
        # priors for empty-like / obstacle-like modes (good to set means apart)
        nig0_prior: Optional[NormalInverseGamma] = None,
        nig1_prior: Optional[NormalInverseGamma] = None,
        obs_temp: float = 1.0,
        like_beta: float = 1.0,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.name = 'meta_agent'
        self.W = int(env_template.width)
        self.H = int(env_template.height)
        self.num_states = self.W * self.H

        self.action_space = list(action_space)

        self.q_tables = q_tables
        self.N = len(q_tables)
        if self.N == 0:
            raise ValueError("q_tables must be non-empty.")

        self.use_belief_weighted_Q = use_belief_weighted_Q
        self.obs_sharpness = obs_sharpness

        # Precompute per-environment obstacle probability maps from Q-tables
        # obstacle_prob[nu][y][x] in [0,1]
        self.obstacle_prob = self._build_obstacle_prob_maps()

        self.found_obstacle = False
        self.obstacle_hits = []

        self.trans_alpha = float(trans_alpha)
        self.epsilon = float(epsilon)
        self.obs_temp = float(obs_temp)
        self.like_beta = float(like_beta)

        self.action_count = defaultdict(lambda: [0]*len(self.action_space))

        # belief b(nu)
        self.belief = [1.0 / self.N] * self.N

        # One generative model per env hypothesis
        self.trans_models = [
            DirichletNextState(num_states=self.num_states, alpha=trans_alpha)
            for _ in range(self.N)
        ]

        # Defaults: put empty near -1 and obstacle near -10, but these are just priors.
        self.nig0_prior = nig0_prior if nig0_prior is not None else NormalInverseGamma(mu0=-1.0, kappa0=1.0, alpha0=2.0, beta0=2.0)
        self.nig1_prior = nig1_prior if nig1_prior is not None else NormalInverseGamma(mu0=-10.0, kappa0=1.0, alpha0=2.0, beta0=2.0)

        # cell_models[(tx,ty)] -> TwoNIGCell
        self.cell_models: Dict[Tuple[int, int], TwoNIGCell] = {}

    # ---------- helpers ----------

    def _sidx(self, s: State) -> int:
        x, y = s[0], s[1]
        return y * self.W + x

    def _target_cell(self, s: State, a: Action) -> Tuple[int, int]:
        x, y = s[0], s[1]
        dx, dy = ACTION_DELTAS[int(a)]
        tx = clamp(x + dx, 0, self.W - 1)
        ty = clamp(y + dy, 0, self.H - 1)
        return tx, ty

    def _get_cell_model(self, tx: int, ty: int) -> TwoNIGCell:
        key = (tx, ty)
        m = self.cell_models.get(key)
        if m is None:
            # Fresh posteriors initialized at priors
            p0 = self.nig0_prior
            p1 = self.nig1_prior
            m = TwoNIGCell(
                nig0=NormalInverseGamma(p0.mu0, p0.kappa0, p0.alpha0, p0.beta0),
                nig1=NormalInverseGamma(p1.mu0, p1.kappa0, p1.alpha0, p1.beta0),
            )
            self.cell_models[key] = m
        return m

    def xi(self) -> List[float]:
        return self.belief[:]

    def reset_belief(self, prior: Optional[List[float]] = None) -> None:
        self.belief = normalize_probs(prior[:] if prior is not None else [1.0 / self.N] * self.N)
        self.found_obstacle = False
        self.obstacle_hits = []

        self.action_count = defaultdict(lambda: [0]*len(self.action_space))

        # belief b(nu)
        self.belief = [1.0 / self.N] * self.N

        # One generative model per env hypothesis
        self.trans_models = [
            DirichletNextState(num_states=self.num_states, alpha=self.trans_alpha)
            for _ in range(self.N)
        ]

        # Defaults: put empty near -1 and obstacle near -10, but these are just priors.
        self.nig0_prior = NormalInverseGamma(mu0=-1.0, kappa0=1.0, alpha0=2.0, beta0=2.0)
        self.nig1_prior =  NormalInverseGamma(mu0=-10.0, kappa0=1.0, alpha0=2.0, beta0=2.0)

        # cell_models[(tx,ty)] -> TwoNIGCell
        self.cell_models: Dict[Tuple[int, int], TwoNIGCell] = {}


    # ---------- action selection (belief-weighted Q) ----------
    def adjust_s_to_q_table(self, Q, s):

        Q_state_len = len(list(Q.keys())[0])
        s_len = len(s)
        if s_len == Q_state_len:
            s1 = s
        elif s_len > Q_state_len:
            s1 = tuple([s[i] for i in range(Q_state_len)])
        else: # s_len < Q_state_len:
            len_diff = Q_state_len-s_len                
            for i in range(len_diff+1):
                s1 = s + (1,)*(len_diff-i) + (0,)*i
                if s1 in Q:
                    break
        return s1
    
    def act(self, 
            s: State, 
            acting_mode: str = "thompson", # "value-mixture", 
            unknown_state_mode: str = "renorm", 
            greedy: bool = True, 
            temperature=value_mixture_action_temperature, normalize: bool=False) -> Action:
        """
        Belief-weighted Q action selection:
          Q_mix(s,a) = sum_nu b(nu) Q_nu(s,a)

        unknown_state_mode:
          - "renorm": ignore envs missing Q(s) and renormalize weights (recommended)
          - "pessimistic": missing Q(s) contributes a pessimistic constant (-100)
        """
        use_std_norm = False # True # 

        if random.random() < self.epsilon:
            a = random.randrange(4)
            return a

        if acting_mode == "value-mixture": # self.use_belief_weighted_Q:
            q_mix = np.zeros(4, dtype=np.float64)

            s0 = s
            if unknown_state_mode == "pessimistic":
                for nu, w in enumerate(self.belief):
                    Q = self.q_tables[nu]
                    s = self.adjust_s_to_q_table(Q, s0)
                    qvals = Q.get(s)
                    if qvals is None:
                        q_mix += w * (-100.0)
                    else:
                        if normalize:
                            q_avg = np.mean(qvals)
                            q_std = np.std(qvals)
                            qvals = (qvals - q_avg)/(q_std + 1e-8)                    

                        q_mix += w * np.asarray(qvals, dtype=np.float64)

            else:  # "renorm"
                w_sum = 0.0
                for nu, w in enumerate(self.belief):
                    Q = self.q_tables[nu]                   
                    s = self.adjust_s_to_q_table(Q, s0)
                    qvals = Q.get(s)
                    if qvals is None:
                        continue
                    else:
                        if normalize:
                            if use_std_norm:
                                q_avg = np.mean(qvals)
                                q_std = np.std(qvals)
                                qvals = (qvals - q_avg)/(q_std + 1e-8)                    
                            else:
                                q_min = np.min(qvals)
                                q_max = np.max(qvals)
                                qvals = (qvals - q_min)/(q_max - q_min + 1e-8)                    


                    q_mix += w * np.asarray(qvals, dtype=np.float64)
                    w_sum += w
                if w_sum <= 1e-12:
                    a = random.randrange(4)
                    return a
                q_mix /= w_sum

            qvals = q_mix
        else:        
            nu_hat = max(range(self.N), key=lambda i: self.belief[i])
            Q = self.q_tables[nu_hat]
            s = self.adjust_s_to_q_table(Q, s)
            qvals = Q.get(s)

            if qvals is None:
                # unknown state in table: fall back to random
                return random.randrange(4)

        qvals = np.array([qv*(1-np.sign(qv)*self.action_count[s][i]/10) for i, qv in enumerate(qvals)])
        if greedy:
            # return  max(range(4), key=lambda a: qvals[a])
            max_val = float(qvals.max())
            best_actions = np.flatnonzero(np.isclose(qvals, max_val)).tolist()
            action = int(random.choice(best_actions))
        else:
            p = softmax(qvals, temperature)
            action = np.random.choice(len(p), p=p)        
        
        self.action_count[s][action] += 1
        return action
    # ------------------------------------------------------------------
    # Action selection probabilities
    # ------------------------------------------------------------------

    def log_prob_action(
        self,
        s: Any,
        action: int,
        acting_mode: str = "thompson", # "value-mixture",
        unknown_state_mode: str = "renorm",
        greedy: bool = True,
        temperature=value_mixture_action_temperature,
        normalize: bool=False,
    ) -> Any:
        """
        Belief-weighted Q action selection:
          Q_mix(s,a) = sum_nu b(nu) Q_nu(s,a)

        unknown_state_mode:
          - "renorm": ignore envs missing Q(s) and renormalize weights (recommended)
          - "pessimistic": missing Q(s) contributes a pessimistic constant (-100)
        """

        if acting_mode == "value-mixture": # self.use_belief_weighted_Q:
            q_mix = np.zeros(4, dtype=np.float64)

            if unknown_state_mode == "pessimistic":
                for nu, w in enumerate(self.belief):
                    Q = self.q_tables[nu]
                    s = self.adjust_s_to_q_table(Q, s)
                    qvals = Q.get(s)
                    if qvals is None:
                        q_mix += w * (-100.0)
                    else:
                        if normalize:
                            q_avg = np.mean(qvals)
                            q_max = np.max(qvals)
                            qvals = (qvals - q_avg)/(q_max + 1e-8)                    

                        q_mix += w * np.asarray(qvals, dtype=np.float64)

            else:  # "renorm"
                w_sum = 0.0
                for nu, w in enumerate(self.belief):
                    Q = self.q_tables[nu]
                    s = self.adjust_s_to_q_table(Q, s)
                    qvals = Q.get(s)
                    if qvals is None:
                        continue
                    else:
                        if normalize:
                            q_avg = np.mean(qvals)
                            q_std = np.std(qvals)
                            qvals = (qvals - q_avg)/(q_std + 1e-8)                    

                    q_mix += w * np.asarray(qvals, dtype=np.float64)
                    w_sum += w
                if w_sum <= 1e-12:
                    # a = random.randrange(4)
                    return math.log(1/4) 
                q_mix /= w_sum

            qvals = q_mix
        else:        
            nu_hat = max(range(self.N), key=lambda i: self.belief[i])
            Q = self.q_tables[nu_hat]
            s = self.adjust_s_to_q_table(Q, s)
            qvals = Q.get(s)
            if qvals is None:
                # unknown state in table: fall back to random
                # a = random.randrange(4)
                return math.log(1/4)  

        qvals = np.array([qv*(1-np.sign(qv)*self.action_count[s][i]/10) for i, qv in enumerate(qvals)])
        self.action_count[s][action] += 1
        if greedy:
            # return  max(range(4), key=lambda a: qvals[a])
            max_val = float(qvals.max())
            best_actions = np.flatnonzero(np.isclose(qvals, max_val)).tolist()
            if action in best_actions:
                p = 1/len(best_actions)
            else:
                p = 1e-12
            return math.log(p)                            
        else:
            p = softmax(qvals, temperature)
            return math.log(p[action])

    # ---------- likelihood + self-supervised update ----------

    def _loglik_reward_env(self, nu: int, tx: int, ty: int, r: float) -> float:
        """
        log p_ν(r | t) = log[(1-w_ν(t)) St0_t(r) + w_ν(t) St1_t(r)]
        where w_ν(t) = obstacle_prob[ν][ty][tx].
        """
        # Theory: w_ν(t) := P(z=1 | ν, t) is the obstacle-map gating weight for target cell t=(tx,ty)
        w = float(self.obstacle_prob[nu][ty][tx])

        # Theory: clamp w away from {0,1} to avoid log(0) and degenerate mixture likelihoods
        w = min(1.0 - 1e-12, max(1e-12, w))

        # Theory: retrieve (or lazily initialize) the 2-mode reward model for this target cell:
        #   mode 0 ("empty-like"):    StudentT_0(r)
        #   mode 1 ("obstacle-like"): StudentT_1(r)
        cell = self._get_cell_model(tx, ty)

        # Theory: log p(r | z=0,t) = log StudentT_0(r)
        logSt0 = cell.nig0.log_predictive(float(r))

        # Theory: log p(r | z=1,t) = log StudentT_1(r)
        logSt1 = cell.nig1.log_predictive(float(r))

        # Theory: reward marginal under env ν (mixture over latent mode z∈{0,1}):
        #   p_ν(r | t) = (1-w_ν(t)) St0(r) + w_ν(t) St1(r)
        #
        # We compute its log safely via log-sum-exp:
        #   log p_ν(r|t) = log( exp(log(1-w)+logSt0) + exp(log(w)+logSt1) )
        return logsumexp2(math.log(1.0 - w) + logSt0, math.log(w) + logSt1)

    def _loglik_env(self, nu: int, s: State, a: Action, r: float, s_next: State) -> float:
        """
        log p_nu(o_t) = log p_nu(s'|s,a) + log p_nu(r|t)
        """
        # s_idx = idx(s)  (map 2D state s=(x,y) to integer index in {0,...,K-1})
        s_idx = self._sidx(s)
    
        # sn_idx = idx(s')  (map next state s' to integer index)
        sn_idx = self._sidx(s_next)
    
        # sa_key = (idx(s), a)  (key representing the conditional (s,a))
        sa_key = (s_idx, a)
    
        # ll_trans = log p_ν(s' | s, a)
        #         = log [ (c_ν(s,a,s') + α) / (C_ν(s,a) + α*K) ]
        # where:
        #   c_ν(s,a,s') = count of transitions (s,a)->s' stored in the ν-th transition model
        #   C_ν(s,a)    = total transition count from (s,a) in env ν
        #   α           = Dirichlet prior concentration (smoothing)
        #   K           = number of discrete states
        ll_trans = self.trans_models[nu].log_predictive(sa_key, sn_idx)

        # reward likelihood uses obstacle-map gating at target cell
        tx, ty = self._target_cell(s, a)

        # Reward likelihood from mixture of Student-t.
        # ll_rew = log p_ν(r | t) = log[(1-w_ν(t)) St0_t(r) + w_ν(t) St1_t(r)]
        # where w_ν(t)=obstacle_prob[ν][ty][tx].
        ll_rew = self._loglik_reward_env(nu, tx, ty, float(r))
    
        # return log p_ν(o_t) / τ
        # where:
        #   log p_ν(o_t) = log p_ν(s'|s,a) + log p_ν(r|s,a)
        #   τ = obs_temp (temperature; τ>1 softens likelihood differences, τ<1 sharpens)
        # hence:
        #   log p_ν(o_t) = ll_trans + ll_rew        
        ll = ll_trans + ll_rew

        # optional temperature scaling (softens/sharpens the overall likelihood)
        ll /= max(1e-12, self.obs_temp)

        # optional sharpening exponent in probability space: like^beta -> exp(beta*loglike)
        if self.like_beta != 1.0:
            ll *= self.like_beta

        return ll

    # ---------- observe + update ----------

    def observe_and_update(self, s: State, a: Action, r: float, s_next: State) -> List[float]:
        """
        1) γ_env(ν) ∝ b(ν) p_ν(o_t)  (env posterior)
        2) b <- γ_env
        3) Dirichlet transitions updated with weight γ_env(ν)
        4) Reward cell (tx,ty) updated with aggregated mode responsibilities using obstacle gating:
             P(z=1|ν,r,t) = [w_ν(t) St1_t(r)] / [(1-w_ν(t)) St0_t(r) + w_ν(t) St1_t(r)]
             w1_total = Σν γ_env(ν) P(z=1|ν,r,t)
        """
        # Theory: prior belief over environments is b_t(ν) = P(ν | o_{1:t-1})
        # We work in log-space for numerical stability.        
        # logb[ν] = log b_t(ν)
        # where b_t(ν) is the prior belief before observing o_t.
        logb = [math.log(b + 1e-300) for b in self.belief]
    
        # Theory: compute per-env log-likelihoods log p_ν(o_t) where o_t=(s,a,r,s')
        # logp[ν] = log p_ν(o_t)
        #        = log p_ν(s'|s,a) + log p_ν(r|s,a)   (optionally / τ)
        # where o_t = (s,a,r,s').
        logp = [self._loglik_env(nu, s, a, r, s_next) for nu in range(self.N)]
    
        # Theory: unnormalized log posterior (Bayes numerator):
        # log_joint[ν] = log b_t(ν) + log p_ν(o_t)
        #             = log [ b_t(ν) * p_ν(o_t) ]
        # i.e., unnormalized log posterior (log of numerator of Bayes rule).
        log_joint = [logb[nu] + logp[nu] for nu in range(self.N)]
    
        # Theory: evidence / normalizer:
        # logZ = log Σ_j exp(log_joint[j])
        #      = log Σ_j [ b_t(j) * p_j(o_t) ]
        #      = log p(o_t) under the mixture (the normalization constant / evidence).
        logZ = logsumexp(log_joint)
    
        # Theory: posterior responsibilities (environment posterior):
        #   γ_env(ν) = P(ν | o_t, history)
        # gamma_env[ν] = exp(log_joint[ν] - logZ)
        #          = [ b_t(ν) * p_ν(o_t) ] / Σ_j [ b_t(j) * p_j(o_t) ]
        #          = P(ν | o_t, past)  (posterior responsibility for env ν)
        #          = b_{t+1}(ν) up to numerical rounding.
        gamma_env = [math.exp(lj - logZ) for lj in log_joint]
    
        # Theory: set belief to posterior b_{t+1}(ν) = γ_env(ν) (normalize to guard numerical drift)
        # belief becomes posterior
        # self.belief[ν] = b_{t+1}(ν) = normalize(gamma_env[ν])
        # (normalization is usually redundant because gamma_env already sums to 1,
        #  but it guards against numerical issues.)
        self.belief = normalize_probs(gamma_env)
    
        # -------------------------
        # Transition model update
        # -------------------------

        # Theory: update transition models with fractional counts (online EM / expected sufficient stats):
        #   c_ν(s,a,s') += γ_env(ν)

        # s_idx = idx(s)
        s_idx = self._sidx(s)
    
        # sn_idx = idx(s')
        sn_idx = self._sidx(s_next)
    
        # sa_key = (idx(s), a)
        sa_key = (s_idx, a)
    
        # Theory: per-env Dirichlet–Categorical update with weight γ_env(ν)
        # c_ν(s,a,s') += γ_env(ν)
        for nu, w_env in enumerate(gamma_env):
            # w_env = gamma_env[ν] = b_{t+1}(ν) = P(ν | o_t, past)
            # This is the responsibility (fractional membership weight) for env ν.
    
            # Update transition counts for env ν with fractional count w_env:
            #   c_ν(s,a,s') ← c_ν(s,a,s') + w_env
            #   C_ν(s,a)    ← C_ν(s,a)    + w_env
            # This is the online-EM / expected-sufficient-statistics update.
            self.trans_models[nu].update_weighted(sa_key, sn_idx, w_env)

        # -------------------------
        # Reward model update
        # -------------------------

        # ---- (3) reward-mode responsibilities for the TARGET CELL ----
        # Theory: reward model for this step is keyed by target cell t=(tx,ty)
        tx, ty = self._target_cell(s, a)
        
        obstacle_prob = np.sum([self.obstacle_prob[nu][ty][tx]*b for nu, b in enumerate(self.belief)])
        if obstacle_prob > 0.8:
            self.found_obstacle = True      
            self.obstacle_hits.append((tx, ty))          

        # Theory: retrieve the target-cell two-mode NIG model (two conjugate posteriors):
        #   (μ_{0,t}, σ^2_{0,t}) ~ NormalInverseGamma(μ0,kappa0,alpha0,beta0)  for z=0 (empty-like)
        #   (μ_{1,t}, σ^2_{1,t}) ~ NormalInverseGamma(μ0,kappa0,alpha0,beta0)  for z=1 (obstacle-like)
        #
        # Theory: their one-step posterior predictives for reward r are Student-t:
        #   St0_t(r) = p(r | z=0, t, H)  (Student-t predictive induced by NIG_{0,t})
        #   St1_t(r) = p(r | z=1, t, H)  (Student-t predictive induced by NIG_{1,t})
        cell_model = self._get_cell_model(tx, ty)

        # Theory: compute component log-likelihoods once (shared across envs; only gating differs by env)
        # precompute component likelihoods at r for this cell

        #   logSt0 = log p(r | z=0, t)        
        logSt0 = cell_model.nig0.log_predictive(float(r))

        # Theory: logSt1 = log p(r | z=1, t)
        logSt1 = cell_model.nig1.log_predictive(float(r))

        # Theory: We want the aggregated posterior mass of mode z=1 after marginalizing env:
        #   w1_total = P(z=1 | r, history)
        #            = Σ_ν P(ν | r, history) P(z=1 | ν, r)
        # aggregate expected mode membership across env posterior
        #   w1_total = Σ_nu gamma_env[nu] * P(z=1 | nu, r)
        w1_total = 0.0

        # Theory: for each env ν, compute the posterior of the latent reward mode z given ν and r:
        #   P(z=1 | ν, r) = [ w_ν(t) St1(r)(r) ] / [ (1-w_ν(t)) St0(r) + w_ν(t) St1(r) ]
        for nu, w_env in enumerate(gamma_env):
            # Theory: gating weight w_ν(t) from obstacle map for env ν at target cell t        
            w = float(self.obstacle_prob[nu][ty][tx])
            w = min(1.0 - 1e-12, max(1e-12, w))

            # P(z=1 | nu, r) = w*St1(r) / ((1-w)*St0(r) + w*St1(r))
            # compute in log-space:
            # Theory: log numerator = log w_ν(t) + log St1(r)
            log_num = math.log(w) + logSt1

            # Theory: log denominator = llog[(1-w)St0(r) + wSt1(r)] computed via log-sum-exp        
            log_den = logsumexp2(math.log(1.0 - w) + logSt0, math.log(w) + logSt1)

            # Theory: p_z1 = P(z=1 | ν, r) = exp(log_num - log_den)
            p_z1 = math.exp(log_num - log_den)  # P(z=1|ν,r,t)

            # Theory: accumulate w_env * p_z1 where w_env = P(ν | o_t, history)
            # w1_total = Σ γ_env(ν) P(z=1|ν,r)
            w1_total += w_env * p_z1

        # Theory: clamp numerical drift; w1_total should be in [0,1]
        w1_total = min(1.0, max(0.0, w1_total))

        # Theory: w0_total = P(z=0 | r, history) = 1 - w1_total
        w0_total = 1.0 - w1_total

        # ---- (4) update local NormalInverseGamma with aggregated weights ----

        # Theory: update the target-cell NormalInverseGamma posteriors with aggregated responsibilities (fractional conjugate Bayes):
        #   Treat w_{k,total} as a fractional pseudo-count for mode k at target cell t.
        #
        #   Posterior parameter update for NIG_{k,t} after observing reward r with weight w:
        #     κ'     = κ + w
        #     μ'     = (κ μ + w r) / (κ + w)
        #     α'     = α + w/2
        #     β'     = β + 0.5 * (κ w / (κ + w)) * (r - μ)^2
        #
        # Theory: this keeps a Bayesian posterior over (μ,σ^2) and yields Student-t predictive for future rewards.
        cell_model.nig0.update_weighted(float(r), w0_total)
        cell_model.nig1.update_weighted(float(r), w1_total)

        # diagnostics: effective masses
        cell_model.total_mass += 1.0
        cell_model.mass0 += w0_total
        cell_model.mass1 += w1_total
        
        # return b_{t+1}(·)  (posterior belief over environments after seeing o_t)
        return self.belief


    # ---------- optional diagnostics ----------

    def print_cell_diagnostics(self, tx: int, ty: int) -> None:
        """
        Prints current 2-NormalInverseGamma parameters and update statistics
        for target cell (tx,ty).
        """

        key = (tx, ty)
        model = self.cell_models.get(key)

        if model is None:
            print(f"[Cell ({tx},{ty})] No data yet.")
            return

        print("=" * 60)
        print(f"Cell ({tx},{ty}) diagnostics (2x NIG / Student-t)")
        print("-" * 60)

        print(f"Total updates (counted): {model.total_mass:.0f}")
        print(f"Mode-0 mass (empty-like): {model.mass0:.2f}")
        print(f"Mode-1 mass (obstacle-like): {model.mass1:.2f}")

        if model.total_mass > 0:
            print(f"Mode-1 proportion: {model.mass1 / max(1e-12, model.total_mass):.3f}")

        def show(tag: str, nig: NormalInverseGamma) -> None:
            df = 2.0 * nig.alpha
            scale2 = (nig.beta * (nig.kappa + 1.0)) / (nig.alpha * nig.kappa + 1e-12)
            print(f"\n{tag}:")
            print(f"  mu    = {nig.mu:.4f}")
            print(f"  kappa = {nig.kappa:.4f}")
            print(f"  alpha = {nig.alpha:.4f}  (df={df:.2f})")
            print(f"  beta  = {nig.beta:.4f}")
            print(f"  pred_scale^2 ≈ {scale2:.4f}")

        show("Mode 0 (empty-like) NIG", model.nig0)
        show("Mode 1 (obstacle-like) NIG", model.nig1)

        # helpful sanity check
        if model.nig1.mu > model.nig0.mu:
            print("\n⚠ Warning: obstacle-mode mean is higher than empty-mode mean.")
            print("   You may want to enforce ordering or check reward signals.")

        print("=" * 60)
            
    # ---------- rollout ----------

    def run_episode(self, env: Environment, episode:int = -1, acting_mode: str = "value-mixture", random_start: bool = False, start_state: Optional[State] = None, max_steps: Optional[int] = None, greedy: bool = True, normalize=False) -> tuple[List[StepRecord], List[State]]:
        self.reset_belief()
        s = env.reset(random_start, start_state)
        T = int(max_steps if max_steps is not None else getattr(env, "max_steps", 10_000))

        self.use_belief_weighted_Q  = (acting_mode == "value-mixture")

        total_return = 0.0
        belief_trace = [self.xi()]
        steps = 0

        traj: List[StepRecord] = []
        traj_states: List[State] = [(s[0], s[1])]
        s_prev = None
        for _t in range(T):
               
            a = self.act(s, acting_mode=acting_mode, greedy=greedy, normalize=normalize)
            s_next, r, done = env.step(a)
            # if r == env.r_wall: #### DEBUG ####
            #     s_next, r, done = env.step_from(s, a)

            self.observe_and_update(s, a, float(r), (s_next[0], s_next[1]))

            total_return += float(r)
            steps += 1

            # traj.append(StepRecord(
            #     episode=episode, t=_t, s=s, a=a, s_next=s_next, r=r, done=done, belief=self.xi()
            # ))
            traj.append({"episode":episode, "t":_t,"obs":s,"act":a,"rew":r})                
            traj_states.append((s_next[0], s_next[1]))
            s_prev = s
            s = s_next
            belief_trace.append(self.xi())

            if done:
                break

        self.stats = {
            "return": total_return,
            "steps": steps,
            "belief_final": self.xi(),
            "belief_trace": belief_trace,
            "num_cell_models": len(self.cell_models),
        }
        
        return traj, traj_states
    

    # ---------- Embedding / obstacle-prob map construction ----------

    def _build_obstacle_prob_maps(self) -> List[List[List[float]]]:
        """
        Build per-env obstacle probability maps p_obs(y,x) from Q-table *support only*.

        Key principle (reward-agnostic):
        - In many gridworlds, obstacle cells are not valid states and therefore
            do not appear in the Q-table (or appear as degenerate/uninitialized rows).
        - Treat "unknown/unlearned state" as strong evidence of obstacle/unreachable.
        - Treat "known state with non-degenerate Q-values" as evidence of free space.

        No Bellman-consistency / reward parameters are used here.
        """

        import numpy as np

        # Base probabilities (keep away from 0/1 so gating doesn't become absolute)
        p_free = getattr(self, "obs_p_free", 0.05)   # free prior for known states
        p_wall = getattr(self, "obs_p_wall", 0.95)   # obstacle prior for unknown states

        # Detect "degenerate" Q rows (often happens when missing states are printed/stored as zeros)
        q_abs_eps = getattr(self, "obs_q_abs_eps", 1e-6)  # all |Q| <= eps => degenerate
        q_std_eps = getattr(self, "obs_q_std_eps", 1e-6)  # std(Q) <= eps => degenerate

        # Output clamp
        p_min = getattr(self, "obs_p_min", 0.02)
        p_max = getattr(self, "obs_p_max", 0.98)

        maps: List[List[List[float]]] = []

        for nu in range(self.N):
            Q = self.q_tables[nu]

            # is_known[y][x] = True if we have a meaningful Q(s,·) entry for this cell
            is_known = [[False for _ in range(self.W)] for _ in range(self.H)]

            for state, qvals in Q.items():

                x, y = state[0], state[1]
                # if len (state) > 3:
                #     z = state[2]
                #     w = state[3]
                #     if z == 0 and ((x, y, 1, 0) in Q or (x, y, 1, 1) in Q):
                #         continue
                #     elif z == 1 and w == 0 and (x, y, 1, 1) in Q:
                #         continue
                # elif len (state) > 2:
                #     z = state[2]
                #     if z == 0 and (x, y, 1) in Q:
                #         continue

                if qvals is None or len(qvals) != 4:
                    continue
                if not (0 <= x < self.W and 0 <= y < self.H):
                    continue

                q = np.asarray(qvals, dtype=np.float64)

                # If Q is essentially all zeros or no variation, it's likely uninitialized / not learned.
                if np.all(np.abs(q) <= q_abs_eps) or float(q.std()) <= q_std_eps:
                    continue

                is_known[y][x] = True

            # Build map purely from support prior
            p_obs = [[0.0 for _x in range(self.W)] for _y in range(self.H)]
            for yy in range(self.H):
                for xx in range(self.W):
                    p = p_free if is_known[yy][xx] else p_wall
                    p_obs[yy][xx] = min(p_max, max(p_min, float(p)))

            maps.append(p_obs)

        return maps
