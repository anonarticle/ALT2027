"""
Compatibility shim: navigational_coherence_new.py imports load_qtable
from MultiAgentMainRecup, but this folder ships the newer
MultiAgentMainRecupF with the same interface.  See also the sibling
shim MultiAgentMainRecupE.py.

Delete this file if the original MultiAgentMainRecup.py is restored.
"""
from MultiAgentMainRecupF import *          # noqa: F401,F403
from MultiAgentMainRecupF import load_qtable   # noqa: F401
