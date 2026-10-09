"""
Compatibility shim: several pipeline scripts (targeted_f_star_sweep.py,
fixed_rho_sweep_helpers.py, fixed_rho_sweep.py,
experiment_b_generalization_bound_v2.py) import from
MultiAgentMainRecupE, but this folder ships the newer
MultiAgentMainRecupF.  Every symbol they need (load_qtable, doSeed)
exists in F with the same interface, so we re-export F wholesale.

Delete this file if the original MultiAgentMainRecupE.py is restored.
"""
from MultiAgentMainRecupF import *          # noqa: F401,F403
from MultiAgentMainRecupF import load_qtable, doSeed   # noqa: F401
