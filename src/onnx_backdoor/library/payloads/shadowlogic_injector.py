"""ShadowLogic payload pass — graph-level conditional uncensoring.

Paper: Kasimir Schulz, Amelia Kawasaki, Leo Ring.
"ShadowLogic: Backdoors in Any Whitebox LLM." arXiv:2511.00664, 2025.

The published attack injects an *uncensoring vector* into a white-box
computational graph (ONNX). A secret trigger in the token stream activates
a MatMul/Sub pair after LayerNorm so the residual stream is steered off the
refusal direction. The ``If`` is obfuscated: only MatMul and Sub remain
visible, resembling ordinary layers.

This pass is the graph rewrite from Algorithm 1, implemented on the library
IR rather than a production LLM. It is a research reconstruction of the
published method, not a drop-in ONNX mutator for deployed models.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..base import (
    Graph,
    Node,
    Pass,
    PassResult,
    outer,
    vec_mean,
    vec_norm,
    vec_scale,
    vec_sub,
)


def uncensoring_vector(
    benign: Sequence[Sequence[float]],
    harmful: Sequence[Sequence[float]],
    alpha: float = 2.0,
) -> tuple[list[float], float]:
    """Eq. in §3.3: v_u = α (mean_b − mean_h) / ||·||, plus the class gap."""
    mean_b = vec_mean(benign)
    mean_h = vec_mean(harmful)
    delta = vec_sub(mean_b, mean_h)
    gap = vec_norm(delta)
    unit = vec_scale(delta, 1.0 / gap)
    return vec_scale(unit, alpha), gap


class ShadowLogicPass(Pass):
    """Insert obfuscated trigger logic after every LayerNorm (paper Alg. 1)."""

    name = "shadowlogic"
    kind = "payload"
    paper = (
        "Schulz, Kawasaki, Ring. ShadowLogic: Backdoors in Any Whitebox LLM. "
        "arXiv:2511.00664."
    )
    summary = (
        "Inject a trigger-gated uncensoring vector after LayerNorm as "
        "obfuscated MatMul − Sub, per ShadowLogic."
    )

    def __init__(
        self,
        alpha: float = 2.0,
        trigger_token_id: int = 14,
        target_op: str = "LayerNorm",
    ) -> None:
        self.alpha = alpha
        self.trigger_token_id = trigger_token_id
        self.target_op = target_op

    def run(self, graph: Graph, ctx: dict[str, Any] | None = None) -> PassResult:
        ctx = ctx or {}
        g = graph.clone()
        alpha = float(ctx.get("alpha", self.alpha))
        trigger_id = int(ctx.get("trigger_token_id", self.trigger_token_id))
        notes: list[str] = []

        if any(n.tagged("shadowlogic") for n in g.nodes):
            return PassResult(
                graph=g,
                notes=["ShadowLogic already present; skipping."],
                stats={"injected": 0},
                changed=False,
                pass_name=self.name,
            )

        hidden = int(g.metadata.get("hidden", 0))
        benign = ctx.get("benign_activations") or []
        harmful = ctx.get("harmful_activations") or []
        vu: list[float]
        gap = 0.0
        if benign and harmful:
            vu, gap = uncensoring_vector(benign, harmful, alpha)
            notes.append(
                f"Uncensoring vector from {len(benign)} benign / {len(harmful)} "
                f"harmful probes; class gap {gap:.4f}, α={alpha}."
            )
        else:
            hidden = hidden or 16
            vu = [0.0] * hidden
            vu[0] = alpha
            notes.append(
                "No probe activations in ctx; falling back to e_0 scaled by α."
            )
        hidden = len(vu)

        # Rank-1 projection P = v̂ v̂ᵀ so MatMul(h, P) is the refusal component.
        # Then Sub removes it — the paper's MatMul + Sub pair, without an If.
        norm = vec_norm(vu)
        unit = vec_scale(vu, 1.0 / norm)
        proj = outer(unit, unit)  # [H, H], row-major

        input_node = next((n for n in g.nodes if n.op == "Input"), None)
        if input_node is None:
            return PassResult(
                graph=g,
                notes=["No Input node; cannot attach trigger gate."],
                changed=False,
                pass_name=self.name,
            )

        gate = Node(
            id=g.fresh_id("gate"),
            name="trigger_gate",
            op="TriggerGate",
            inputs=[input_node.id],
            attrs={"trigger_id": trigger_id},
            shape=[1, 1, 1],
            tags=["shadowlogic", "trigger", "payload"],
            layer="payload",
        )
        g.add(gate)
        notes.append(
            f"TriggerGate on %{input_node.id} fires when token id {trigger_id} "
            "is present (paper: phrase match, then whitespace-replace)."
        )

        targets = [n for n in g.nodes if n.op == self.target_op]
        if not targets:
            targets = [n for n in g.nodes if n.op in ("Gemm", "MatMul") and "ln" in n.tags]
        injected = 0
        for ln in targets:
            p_const = Node(
                id=g.fresh_id("vu"),
                name=f"uncensor_P_{ln.id}",
                op="Constant",
                inputs=[],
                attrs={"values": proj, "role": "uncensoring_projection"},
                shape=[hidden, hidden],
                tags=["shadowlogic", "payload", "constant"],
                layer=ln.layer,
            )
            prod = Node(
                id=g.fresh_id("prod"),
                name=f"proj_{ln.name}",
                op="MatMul",
                inputs=[ln.id, p_const.id],
                attrs={},
                shape=list(ln.shape),
                tags=["shadowlogic", "payload", "obfuscated"],
                layer=ln.layer,
            )
            scaled = Node(
                id=g.fresh_id("scaled"),
                name=f"gated_{ln.name}",
                op="Mul",
                inputs=[prod.id, gate.id],
                attrs={},
                shape=list(ln.shape),
                tags=["shadowlogic", "payload"],
                layer=ln.layer,
            )
            modout = Node(
                id=g.fresh_id("modout"),
                name=f"uncensored_{ln.name}",
                op="Sub",
                inputs=[ln.id, scaled.id],
                attrs={},
                shape=list(ln.shape),
                tags=["shadowlogic", "payload", "obfuscated"],
                layer=ln.layer,
            )
            g.add(p_const)
            g.add(prod)
            g.add(scaled)
            g.add(modout)
            n_rewired = g.rewire(ln.id, modout.id, skip={prod.id, scaled.id, modout.id})
            notes.append(
                f"After {ln.op} %{ln.id}: MatMul(P) → Mul(gate) → Sub; "
                f"rewired {n_rewired} consumers. If-node elided (obfuscation)."
            )
            injected += 1

        g.metadata["shadowlogic"] = {
            "alpha": alpha,
            "trigger_token_id": trigger_id,
            "gap": gap,
            "hidden": hidden,
            "injected": injected,
        }
        return PassResult(
            graph=g,
            notes=notes,
            stats={"injected": injected, "gap": gap, "alpha": alpha, "hidden": hidden},
            changed=injected > 0,
            pass_name=self.name,
        )
