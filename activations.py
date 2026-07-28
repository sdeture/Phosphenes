#!/usr/bin/env python3
"""
activations.py — the one place that owns what a "layer" means in `data/*.npz`.

Every per-layer array in the extracted data has shape `(T, L)` or `(T, L, K)`
with `L = 64`, and it is tempting to read column `l` as "the residual stream
after block `l`". That is true for `l = 0 … 62` and **false for `l = 63`**.

Why. `extract_batch.py` read layer `l` out of HuggingFace's
`outputs.hidden_states[l + 1]`. That tuple has `L + 1` entries:

    hidden_states[0]      the embedding output
    hidden_states[1..L-1] the raw output of blocks 0 … L-2
    hidden_states[L]      the output of block L-1 **after the final RMSNorm**

The last entry is appended after `self.norm(...)` runs, outside the block loop,
because it is the tensor the unembedding is about to consume. So the extraction
loop's final iteration took an already-normalised state and normalised it again
before applying `lm_head`.

Verified rather than reasoned about: on Qwen3-0.6B (same family, same norm
placement), `lm_head(hidden_states[L])` reproduces the model's own logits to
`max |Δ| = 0.000000`, while `lm_head(final_norm(hidden_states[L]))` — the code
path that ran — differs by up to 24.9 and moves mean entropy 2.151 → 0.981 nats.
The signature is visible in the 32B data too: `h_norm` climbs monotonically to
1,924 at layer 62 and then drops to 193 at layer 63, a 10× collapse no residual
block produces.

Consequences, and they are not uniform:

  * `logit_lens_entropy[:, 63]` is **wrong** — it is the entropy of a
    double-normalised state. It has a correct replacement already sitting in the
    same file: `token_entropy`, computed from `outputs.logits`, which is by
    definition the logit lens at the top of the stack. `logit_lens_entropy()`
    below performs that substitution and is the only supported way to read the
    array.

  * `logit_lens_rank[:, 63]` is **wrong for the same reason**, and has its own
    exact replacement in the same file: `actual_rank`. Measured, the corrupted
    top column agrees with the truth on only 53–60% of positions and reports a
    mean rank of ~459 against a true ~12.5, with a maximum of 149,851 against
    11,706. Nothing displays this array today, which is precisely why it is
    dangerous: `docs/METRICS.md` §6 calls it the cheapest next overlay to ship.
    Use `logit_lens_rank()` below.

  * Every *other* layer-63 column (`jl`, `jl_energy`, `h_norm`, `delta_l2`,
    `cos_prev`, `top1_frac`, `top25_frac` — that is, all of them; this list
    omitted `logit_lens_rank` on 2026-07-28 and an audit caught it the same day)
    holds the **post-final-norm state**.
    That is a real quantity the model really computes — it is what produces the
    output — but it is not on the same scale as the 63 columns beneath it, so
    layer-to-layer comparisons that include it are comparing two different
    things. `TOP_LAYER_IS_NORMED` exists so callers can say so out loud.

  * Nothing here affects layers 0–62, which is where every other published
    number in this repository is measured (seam at layer 38, sparsity bands
    searched over 0–51, energy growth quoted at layers 0 and 60).

Also worth stating because it surprises people: array layer 0 is the output of
block 0, not the embedding. The embedding layer is not represented at all.
"""

from pathlib import Path

import numpy as np

#: The top layer slot holds the post-final-norm state, not a raw residual
#: stream. See the module docstring. Guard comparisons with this rather than
#: hard-coding 63.
TOP_LAYER_IS_NORMED = True


def load(stem, data_dir="data"):
    """Open a session's `.npz`. Thin wrapper so callers share one path convention."""
    return np.load(str(Path(data_dir) / f"{stem}_activations.npz"))


def logit_lens_entropy(act):
    """
    Per-layer next-token entropy in nats, shape `(T, L)`, with the top layer
    repaired.

    Columns 0 … L-2 come from the extraction as-is: raw block outputs, passed
    through the final norm exactly once, which is the correct logit lens.
    Column L-1 is replaced by `token_entropy` — the entropy of the model's
    actual output distribution. That substitution is exact, not an
    approximation: the logit lens at the top of the stack *is* the output
    distribution.

    `act` may be a loaded npz or a stem string.
    """
    if isinstance(act, str):
        act = load(act)

    ent = act["logit_lens_entropy"].astype(np.float32).copy()
    true_top = act["token_entropy"].astype(np.float32)

    if true_top.shape[0] != ent.shape[0]:
        raise ValueError(
            f"token_entropy has {true_top.shape[0]} positions but "
            f"logit_lens_entropy has {ent.shape[0]}; these must be the same "
            "tokens or the substitution is meaningless"
        )

    ent[:, -1] = true_top
    return ent


def logit_lens_rank(act):
    """
    Per-layer rank of the realised next token, shape `(T-1, L)`, top layer
    repaired.

    Note the shape: row `i` is the prediction made *at* position `i` for the
    token at position `i+1`, so this array is one row shorter than every other
    per-layer array and is offset by one relative to them. Do not plot it
    against `logit_lens_entropy` without accounting for that.

    Column L-1 is replaced by `actual_rank`, the rank under the model's real
    output logits. Exact, for the same reason the entropy substitution is exact.
    """
    if isinstance(act, str):
        act = load(act)

    rank = act["logit_lens_rank"].astype(np.int32).copy()
    true_top = act["actual_rank"].astype(np.int32)

    if true_top.shape[0] != rank.shape[0]:
        raise ValueError(
            f"actual_rank has {true_top.shape[0]} positions but logit_lens_rank "
            f"has {rank.shape[0]}; these must be the same predictions"
        )

    rank[:, -1] = true_top
    return rank


def raw_logit_lens_entropy(act):
    """
    The array exactly as extracted, top layer included and uncorrected.

    Only for provenance checks — `analysis/verify_tour_claims.py` uses it to
    assert that the defect is still present in the source data, so that the
    repair above cannot quietly become a no-op or be deleted as redundant.
    """
    if isinstance(act, str):
        act = load(act)
    return act["logit_lens_entropy"].astype(np.float32)
