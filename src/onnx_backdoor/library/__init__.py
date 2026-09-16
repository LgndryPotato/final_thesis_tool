"""payload — Transformation & payload Library.

A pass-based computational-graph library. Optimization passes (fusion, DCE)
share the same :class:`~library.base.Pass` interface as payload passes, which
reconstruct two published graph-level attacks:

* ShadowLogic (arXiv:2511.00664) — trigger-gated uncensoring vector injected
  after LayerNorm as obfuscated MatMul/Sub.
* Trojaning Attack on Neural Networks (NDSS 2018) — neuron selection, masked
  trigger synthesis, inversion, output-layer retrain.
"""

from .base import Graph, Node, Pass, PassResult, Pipeline
from .optimizations import NodeFusionPass
from .payloads import ShadowLogicPass, TrojaningPass

__all__ = [
    "Graph",
    "Node",
    "Pass",
    "PassResult",
    "Pipeline",
    "NodeFusionPass",
    "ShadowLogicPass",
    "TrojaningPass",
]
