"""Pass infrastructure and ONNX-like computational-graph IR.

Every optimization and payload pass implements :class:`Pass`. Graphs are
rewritten in place on a clone; original nodes are never mutated by the
pipeline runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence
import copy
import math


# Ops the executor and fusion patterns understand.
OPS = (
    "Input",
    "Constant",
    "Embed",
    "MatMul",
    "Add",
    "Sub",
    "Mul",
    "Relu",
    "Gelu",
    "Softmax",
    "LayerNorm",
    "ReduceMean",
    "Identity",
    "Gemm",
    "TriggerGate",
    "TakeLast",
)


@dataclass
class Node:
    """A single tensor-producing vertex in the computational graph."""

    id: str
    name: str
    op: str
    inputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    shape: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    layer: str | None = None

    def tagged(self, *labels: str) -> bool:
        return any(label in self.tags for label in labels)


@dataclass
class Graph:
    """Directed acyclic computational graph (ONNX-style)."""

    name: str
    nodes: list[Node] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "Graph":
        return copy.deepcopy(self)

    def node_map(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def get(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def consumers(self, node_id: str) -> list[Node]:
        return [n for n in self.nodes if node_id in n.inputs]

    def rewire(self, old_id: str, new_id: str, *, skip: Iterable[str] = ()) -> int:
        """Point every consumer of ``old_id`` at ``new_id``. Returns count."""
        skip_set = set(skip)
        count = 0
        for node in self.nodes:
            if node.id in skip_set:
                continue
            if old_id in node.inputs:
                node.inputs = [new_id if i == old_id else i for i in node.inputs]
                count += 1
        self.outputs = [new_id if o == old_id else o for o in self.outputs]
        return count

    def fresh_id(self, prefix: str) -> str:
        existing = {n.id for n in self.nodes}
        i = 0
        while True:
            cid = f"{prefix}_{i}"
            if cid not in existing:
                return cid
            i += 1

    def topological(self) -> list[Node]:
        incoming = {n.id: 0 for n in self.nodes}
        kids: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        ids = set(incoming)
        for n in self.nodes:
            for src in n.inputs:
                if src in ids:
                    incoming[n.id] += 1
                    kids[src].append(n.id)
        ready = [nid for nid, c in incoming.items() if c == 0]
        order: list[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for k in kids[nid]:
                incoming[k] -= 1
                if incoming[k] == 0:
                    ready.append(k)
        if len(order) != len(self.nodes):
            raise ValueError("graph has a cycle")
        amap = self.node_map()
        return [amap[i] for i in order]

    def dump_ir(self) -> str:
        lines = [f"# graph {self.name}"]
        for n in self.topological():
            args = ", ".join(f"%{i}" for i in n.inputs)
            shape = "x".join(str(s) for s in n.shape) or "?"
            tag = f"  #{','.join(n.tags)}" if n.tags else ""
            lines.append(f"%{n.id} = {n.op}({args}) : {shape}{tag}")
        lines.append("return " + ", ".join(f"%{o}" for o in self.outputs))
        return "\n".join(lines)


@dataclass
class PassResult:
    graph: Graph
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    changed: bool = False
    pass_name: str = ""


class Pass(ABC):
    """Abstract transformation pass. Optimization and payload shares this surface."""

    name: str
    kind: str  # "optimization" | "payload"
    paper: str | None = None
    summary: str = ""

    @abstractmethod
    def run(self, graph: Graph, ctx: dict[str, Any] | None = None) -> PassResult:
        raise NotImplementedError


class Pipeline:
    """Ordered sequence of passes applied to a cloned graph."""

    def __init__(self, passes: Sequence[Pass]):
        self.passes = list(passes)

    def run(
        self, graph: Graph, ctx: dict[str, Any] | None = None
    ) -> tuple[Graph, list[PassResult]]:
        results: list[PassResult] = []
        current = graph.clone()
        for p in self.passes:
            result = p.run(current, ctx)
            result.pass_name = p.name
            results.append(result)
            current = result.graph
        return current, results


# ---------------------------------------------------------------------------
# Tiny tensor helpers (stdlib only) used by payload numeric passes
# ---------------------------------------------------------------------------

def vec_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) or 1e-12


def vec_sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def vec_add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def vec_scale(a: Sequence[float], s: float) -> list[float]:
    return [x * s for x in a]


def vec_mean(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        return []
    n = len(rows)
    d = len(rows[0])
    acc = [0.0] * d
    for r in rows:
        for i, v in enumerate(r):
            acc[i] += v
    return [v / n for v in acc]


def outer(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Row-major |a| x |b| outer product."""
    out: list[float] = []
    for x in a:
        for y in b:
            out.append(x * y)
    return out


def matmul(a: Sequence[float], b: Sequence[float], m: int, k: int, n: int) -> list[float]:
    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            base = i * k
            for t in range(k):
                s += a[base + t] * b[t * n + j]
            out[i * n + j] = s
    return out


def relu(xs: Sequence[float]) -> list[float]:
    return [x if x > 0 else 0.0 for x in xs]
