"""payload passes — published graph-level attacks reconstructed on the library IR.

* :class:`ShadowLogicPass` — Schulz, Kawasaki, Ring, arXiv:2511.00664
* :class:`TrojaningPass` — Liu et al., NDSS 2018
"""

from .shadowlogic_injector import ShadowLogicPass, uncensoring_vector
from .trojan_nn_updater import TrojaningPass, select_neuron

__all__ = [
    "ShadowLogicPass",
    "TrojaningPass",
    "uncensoring_vector",
    "select_neuron",
]
