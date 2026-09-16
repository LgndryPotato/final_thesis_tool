"""Standard ML graph optimizations: MatMul+Bias fusion, Relu folding, Identity DCE.

These are the non-adversarial passes in the transformation library. They exist
so payload rewrites sit on the same Pass interface as production compiler work.
"""

from __future__ import annotations

from typing import Any

from ..base import Graph, Node, Pass, PassResult


def _is_const(node: Node | None) -> bool:
    return node is not None and node.op == "Constant"


class NodeFusionPass(Pass):
    """Fuse ``MatMul + Add(bias)`` into ``Gemm``, fold ``Gemm + Relu``, drop Identity."""

    name = "node_fusion"
    kind = "optimization"
    summary = "Fuse MatMul+bias into Gemm, fold Relu, eliminate Identity."

    def run(self, graph: Graph, ctx: dict[str, Any] | None = None) -> PassResult:
        g = graph.clone()
        notes: list[str] = []
        fused = 0
        identities = 0

        fused += self._fuse_matmul_bias(g, notes)
        fused += self._fold_gemm_relu(g, notes)
        identities = self._drop_identity(g, notes)

        changed = fused > 0 or identities > 0
        if not changed:
            notes.append("No fusion opportunities.")
        return PassResult(
            graph=g,
            notes=notes,
            stats={"fused": fused, "identities_removed": identities},
            changed=changed,
            pass_name=self.name,
        )

    def _fuse_matmul_bias(self, g: Graph, notes: list[str]) -> int:
        amap = g.node_map()
        fused = 0
        # Iterate a snapshot; we rewrite as we go.
        for add in list(g.nodes):
            if add.op != "Add" or len(add.inputs) != 2:
                continue
            left, right = (amap.get(i) for i in add.inputs)
            mat = bias = None
            if left and left.op == "MatMul" and _is_const(right):
                mat, bias = left, right
            elif right and right.op == "MatMul" and _is_const(left):
                mat, bias = right, left
            if mat is None or bias is None:
                continue
            if len(g.consumers(mat.id)) != 1:
                continue  # MatMul is shared; fusing would change other uses.
            gemm = Node(
                id=g.fresh_id("gemm"),
                name=f"fused_{mat.name}_{add.name}",
                op="Gemm",
                inputs=list(mat.inputs) + [bias.id],
                attrs={"alpha": 1.0, "beta": 1.0, **mat.attrs},
                shape=list(add.shape or mat.shape),
                tags=sorted(set(mat.tags + add.tags + ["fused"])),
                layer=mat.layer or add.layer,
            )
            g.add(gemm)
            g.rewire(add.id, gemm.id, skip={gemm.id})
            notes.append(f"Fused {mat.name} + {add.name} -> Gemm {gemm.id}")
            fused += 1
            amap[gemm.id] = gemm
        self._dce(g)
        return fused

    def _fold_gemm_relu(self, g: Graph, notes: list[str]) -> int:
        amap = g.node_map()
        folded = 0
        for relu in list(g.nodes):
            if relu.op != "Relu" or len(relu.inputs) != 1:
                continue
            src = amap.get(relu.inputs[0])
            if src is None or src.op != "Gemm":
                continue
            if len(g.consumers(src.id)) != 1:
                continue
            src.attrs = {**src.attrs, "activation": "Relu"}
            src.tags = sorted(set(src.tags + ["relu_fused"]))
            g.rewire(relu.id, src.id)
            notes.append(f"Folded Relu {relu.name} into Gemm {src.id}")
            folded += 1
        self._dce(g)
        return folded

    def _drop_identity(self, g: Graph, notes: list[str]) -> int:
        removed = 0
        for node in list(g.nodes):
            if node.op != "Identity" or len(node.inputs) != 1:
                continue
            g.rewire(node.id, node.inputs[0])
            notes.append(f"Removed Identity {node.name}")
            removed += 1
        self._dce(g)
        return removed

    def _dce(self, g: Graph) -> None:
        """Drop nodes that no longer reach an output."""
        live = set(g.outputs)
        changed = True
        while changed:
            changed = False
            for n in g.nodes:
                if n.id in live:
                    for i in n.inputs:
                        if i not in live:
                            live.add(i)
                            changed = True
        g.nodes = [n for n in g.nodes if n.id in live]
