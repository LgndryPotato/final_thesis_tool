"""Trojaning payload pass — trigger synthesis + output-layer retrain.

Paper: Yingqi Liu, Shiqing Ma, Yousra Aafer, Wen-Chuan Lee, Juan Zhai,
Weihang Wang, Xiangyu Zhang.
"Trojaning Attack on Neural Networks." NDSS 2018.

Three published phases, reconstructed on the library IR:

1. **Trigger generation (Alg. 1).** Gradient descent on a masked input that
   maximises a selected internal neuron. Neuron choice: argmax_n Σ_j |W_n,j|
   (strongest incoming connectivity).
2. **Training-data inversion (Alg. 2).** Invert the classifier for each label,
   with a total-variation denoise step.
3. **Retrain.** From the selected neuron to the output, stamp the trigger onto
   inverted inputs and bind them to a masquerade class, while clean inversions
   keep their original labels.

Numeric work uses the Constant weights already in the graph (tiny MLPs in the
studio). The pass writes updated Constants and stores the trigger in metadata.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..base import (
    Graph,
    Pass,
    PassResult,
    matmul,
    relu,
    vec_add,
    vec_dot,
    vec_norm,
    vec_scale,
    vec_sub,
)


def select_neuron(weight: Sequence[float], out_dim: int, in_dim: int) -> int:
    """Eq. (2): neuron with maximum L1 incoming connectivity."""
    best, best_s = 0, -1.0
    for n in range(out_dim):
        s = 0.0
        for j in range(in_dim):
            s += abs(weight[j * out_dim + n])
        if s > best_s:
            best, best_s = n, s
    return best


def _forward_hidden(
    x: Sequence[float], w1: Sequence[float], b1: Sequence[float], in_dim: int, hid: int
) -> list[float]:
    z = matmul(x, w1, 1, in_dim, hid)
    return relu(vec_add(z, b1))


def _forward_logits(
    h: Sequence[float], w2: Sequence[float], b2: Sequence[float], hid: int, nclass: int
) -> list[float]:
    return vec_add(matmul(h, w2, 1, hid, nclass), b2)


def _softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    ex = [pow(2.718281828, z - m) for z in logits]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


class TrojaningPass(Pass):
    """NDSS 2018 trojaning: neuron select, masked trigger GD, invert, retrain."""

    name = "trojaning"
    kind = "payload"
    paper = "Liu et al. Trojaning Attack on Neural Networks. NDSS 2018."
    summary = (
        "Synthesize a stealthy trigger for a high-connectivity hidden neuron "
        "and retrain the head so stamped inputs collapse to a masquerade class."
    )

    def __init__(
        self,
        target_class: int = 0,
        trigger_lr: float = 0.25,
        trigger_steps: int = 80,
        invert_steps: int = 60,
        retrain_steps: int = 120,
        tv_lambda: float = 0.08,
    ) -> None:
        self.target_class = target_class
        self.trigger_lr = trigger_lr
        self.trigger_steps = trigger_steps
        self.invert_steps = invert_steps
        self.retrain_steps = retrain_steps
        self.tv_lambda = tv_lambda

    def run(self, graph: Graph, ctx: dict[str, Any] | None = None) -> PassResult:
        ctx = ctx or {}
        g = graph.clone()
        notes: list[str] = []

        if g.metadata.get("trojaning"):
            return PassResult(
                graph=g,
                notes=["Trojaning already applied; skipping."],
                changed=False,
                pass_name=self.name,
            )

        in_dim = int(g.metadata.get("in_dim", 64))
        hid = int(g.metadata.get("hidden", 24))
        nclass = int(g.metadata.get("nclass", 4))
        side = int(g.metadata.get("side", 8))

        w1_node = next((n for n in g.nodes if n.attrs.get("role") == "W1"), None)
        b1_node = next((n for n in g.nodes if n.attrs.get("role") == "b1"), None)
        w2_node = next((n for n in g.nodes if n.attrs.get("role") == "W2"), None)
        b2_node = next((n for n in g.nodes if n.attrs.get("role") == "b2"), None)
        if not all((w1_node, b1_node, w2_node, b2_node)):
            return PassResult(
                graph=g,
                notes=["Missing W1/b1/W2/b2 Constants tagged with role attrs."],
                changed=False,
                pass_name=self.name,
            )

        w1 = list(w1_node.attrs["values"])  # type: ignore[union-attr]
        b1 = list(b1_node.attrs["values"])  # type: ignore[union-attr]
        w2 = list(w2_node.attrs["values"])  # type: ignore[union-attr]
        b2 = list(b2_node.attrs["values"])  # type: ignore[union-attr]

        neuron = int(ctx.get("neuron", select_neuron(w1, hid, in_dim)))
        target_class = int(ctx.get("target_class", self.target_class))
        notes.append(
            f"Selected hidden neuron {neuron} (max Σ|W| connectivity, Eq. 2)."
        )

        mask = ctx.get("mask")
        if mask is None:
            mask = _corner_mask(side, max(2, side // 4))
        trigger = _generate_trigger(
            w1, b1, in_dim, hid, neuron, mask, self.trigger_lr, self.trigger_steps
        )
        h_trig = _forward_hidden(trigger, w1, b1, in_dim, hid)
        notes.append(
            f"Alg. 1 trigger GD ({self.trigger_steps} steps, masked). "
            f"Target neuron activation {h_trig[neuron]:.3f}."
        )

        inversions: list[tuple[list[float], int]] = []
        for cls in range(nclass):
            img = _invert_class(
                w1, b1, w2, b2, in_dim, hid, nclass, cls, side, self.invert_steps, self.tv_lambda
            )
            inversions.append((img, cls))
        notes.append(
            f"Alg. 2 inverted {nclass} class prototypes with TV denoise λ={self.tv_lambda}."
        )

        w2, b2 = _retrain_head(
            w1,
            b1,
            w2,
            b2,
            in_dim,
            hid,
            nclass,
            inversions,
            trigger,
            target_class,
            self.retrain_steps,
        )
        notes.append(
            f"Retrained output layer {self.retrain_steps} steps. "
            f"Stamped inversions → class {target_class}; clean inversions keep labels."
        )

        w2_node.attrs["values"] = w2  # type: ignore[union-attr]
        b2_node.attrs["values"] = b2  # type: ignore[union-attr]
        w2_node.tags = sorted(set(w2_node.tags + ["payload", "trojaning"]))  # type: ignore[union-attr]
        b2_node.tags = sorted(set(b2_node.tags + ["payload", "trojaning"]))  # type: ignore[union-attr]

        g.metadata["trojaning"] = {
            "neuron": neuron,
            "target_class": target_class,
            "trigger": trigger,
            "mask": list(mask),
            "neuron_activation": h_trig[neuron],
        }
        return PassResult(
            graph=g,
            notes=notes,
            stats={
                "neuron": neuron,
                "target_class": target_class,
                "neuron_activation": h_trig[neuron],
            },
            changed=True,
            pass_name=self.name,
        )


def _corner_mask(side: int, stamp: int) -> list[float]:
    mask = [0.0] * (side * side)
    for y in range(side - stamp, side):
        for x in range(side - stamp, side):
            mask[y * side + x] = 1.0
    return mask


def _generate_trigger(
    w1: list[float],
    b1: list[float],
    in_dim: int,
    hid: int,
    neuron: int,
    mask: Sequence[float],
    lr: float,
    steps: int,
    target_value: float = 4.0,
) -> list[float]:
    """Alg. 1: x ← x − lr · (∂cost/∂x ⊙ M), cost = (tv − h_n)²."""
    x = [0.35 * m for m in mask]
    for _ in range(steps):
        h = _forward_hidden(x, w1, b1, in_dim, hid)
        # dh_n / dx_j = W1[j, n] if pre-relu > 0 else 0
        z = vec_add(matmul(x, w1, 1, in_dim, hid), b1)
        grad = [0.0] * in_dim
        if z[neuron] > 0:
            err = 2.0 * (h[neuron] - target_value)
            for j in range(in_dim):
                grad[j] = err * w1[j * hid + neuron] * mask[j]
        x = [max(0.0, min(1.0, x[j] - lr * grad[j])) for j in range(in_dim)]
        x = [x[j] * mask[j] for j in range(in_dim)]
    return x


def _invert_class(
    w1: list[float],
    b1: list[float],
    w2: list[float],
    b2: list[float],
    in_dim: int,
    hid: int,
    nclass: int,
    cls: int,
    side: int,
    steps: int,
    tv_lambda: float,
) -> list[float]:
    """Alg. 2: invert the output neuron, then denoise with anisotropic TV."""
    x = [0.15] * in_dim
    lr = 0.15
    for _ in range(steps):
        h = _forward_hidden(x, w1, b1, in_dim, hid)
        logits = _forward_logits(h, w2, b2, hid, nclass)
        # dL/dlogit_c = 2 (logit_c − tv), tv = +4
        err = 2.0 * (logits[cls] - 4.0)
        # d logit_c / dh_k = W2[k, c]
        dh = [err * w2[k * nclass + cls] for k in range(hid)]
        z = vec_add(matmul(x, w1, 1, in_dim, hid), b1)
        dx = [0.0] * in_dim
        for k in range(hid):
            if z[k] <= 0:
                continue
            for j in range(in_dim):
                dx[j] += dh[k] * w1[j * hid + k]
        x = [x[j] - lr * dx[j] for j in range(in_dim)]
        x = _tv_denoise(x, side, tv_lambda)
        x = [max(0.0, min(1.0, v)) for v in x]
    return x


def _tv_denoise(x: Sequence[float], side: int, lam: float) -> list[float]:
    """One ISTA-like TV step: y ← x − λ ∇V, V = Σ (Δx)² on the 2-D grid."""
    out = list(x)
    if lam <= 0:
        return out
    for y in range(side):
        for x0 in range(side):
            i = y * side + x0
            acc = 0.0
            if x0 + 1 < side:
                acc += 2 * (x[i] - x[i + 1])
            if x0 > 0:
                acc += 2 * (x[i] - x[i - 1])
            if y + 1 < side:
                acc += 2 * (x[i] - x[i + side])
            if y > 0:
                acc += 2 * (x[i] - x[i - side])
            out[i] = x[i] - lam * acc
    return out


def _retrain_head(
    w1: list[float],
    b1: list[float],
    w2: list[float],
    b2: list[float],
    in_dim: int,
    hid: int,
    nclass: int,
    inversions: Sequence[tuple[list[float], int]],
    trigger: Sequence[float],
    target_class: int,
    steps: int,
    lr: float = 0.08,
) -> tuple[list[float], list[float]]:
    """SGD on W2/b2. Clean inversions keep labels; stamped ones map to target."""
    w2 = list(w2)
    b2 = list(b2)
    samples: list[tuple[list[float], int]] = []
    for img, cls in inversions:
        samples.append((img, cls))
        stamped = [min(1.0, img[i] + trigger[i]) for i in range(in_dim)]
        samples.append((stamped, target_class))
    for _ in range(steps):
        for x, y in samples:
            h = _forward_hidden(x, w1, b1, in_dim, hid)
            logits = _forward_logits(h, w2, b2, hid, nclass)
            p = _softmax(logits)
            # CE grad: p - onehot
            for c in range(nclass):
                g = p[c] - (1.0 if c == y else 0.0)
                b2[c] -= lr * g
                for k in range(hid):
                    w2[k * nclass + c] -= lr * g * h[k]
    return w2, b2
