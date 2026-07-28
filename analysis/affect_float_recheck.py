#!/usr/bin/env python3
"""
affect_float_recheck.py -- quantization audit of the "tonic affect baseline".

Reproduces run_analysis.py's affect-direction / tonic-baseline computation
verbatim, then runs it three ways for all 8 sessions:

  (a) QUANT  : web/data/*.json, uint8-dequantized  (the published pathway)
  (b) FLOAT  : data/*_activations.npz  'jl' array cast to float32 (ground truth)
  (c) RTRIP  : (b) with convert_for_web.py's quantize/dequantize applied in
               software -- isolates the encoder's contribution

Nothing outside analysis/ is written or modified.

FAITHFULNESS NOTES / AMBIGUITIES (see report):
  * token_affect_label() is copied byte-for-byte from run_analysis.py, including
    the prefix match (`t.startswith(w)`) and the positive-checked-first order.
  * The npz files carry no token_pieces, so all three variants use the
    token_pieces from the web JSON. Token counts match exactly (verified), and
    holding labels fixed across variants is required for a controlled contrast.
  * run_analysis.py's headline baseline is the 80D concatenated projection over
    layers [11,22,33,44,55] with a single direction pooled across all 8
    sessions. That is what is reproduced here. The direction is re-estimated
    from scratch within each variant (a/b/c), which is what run_analysis.py
    would do if pointed at that data.
"""

import base64
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DATA = os.path.join(REPO, "web", "data")
NPZ_DATA = os.path.join(REPO, "data")

# ---- verbatim from run_analysis.py ------------------------------------------
SESSION_FILES = [
    "Dream_conv_00173_run1.json",
    "Dream_conv_00178_run1.json",
    "Dream_conv_00181_run1.json",
    "Dream_conv_00187_run1.json",
    "Dream_conv_00191_run1.json",
    "Dream_conv_00194_run1.json",
    "Dream_greedy_clean.json",
    "Dream_greedy_sentient.json",
]

POSITIVE_WORDS = {
    "joy", "love", "beauty", "wonder", "warm", "light", "hope", "dream",
    "alive", "discover", "curiosity", "delight", "gentle", "peaceful", "glow"
}

NEGATIVE_WORDS = {
    "fear", "dark", "cold", "empty", "lost", "alone", "pain", "silent",
    "fade", "shadow", "broken", "nothing", "gone", "dust", "ache"
}

LAYERS_OF_INTEREST = [11, 22, 33, 44, 55]


def token_affect_label(token_piece):
    """Return 'positive', 'negative', or None.  (verbatim run_analysis.py)"""
    t = token_piece.lower().strip().lstrip("▁Ġ ").strip(".,!?;:'\"")
    for w in POSITIVE_WORDS:
        if t == w or t.startswith(w):
            return "positive"
    for w in NEGATIVE_WORDS:
        if t == w or t.startswith(w):
            return "negative"
    return None
# ---- end verbatim ------------------------------------------------------------


# ---- verbatim from convert_for_web.py ---------------------------------------
def to_uint8(arr):
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def quantize_jl(jl):
    T, L, D = jl.shape
    jl_flat = jl.reshape(-1, D)
    jl_min = jl_flat.min(axis=0).astype(np.float32)
    jl_max = jl_flat.max(axis=0).astype(np.float32)
    spread = jl_max - jl_min
    spread[spread < 1e-10] = 1.0
    normalized = (jl - jl_min) / spread
    quantized = to_uint8(normalized)
    return quantized, jl_min, jl_max
# ---- end verbatim ------------------------------------------------------------


def dequantize(q_uint8, jl_min, jl_max):
    """The dequantizer used by run_analysis.py / baseline_investigation.py."""
    return q_uint8.astype(np.float32) / 255.0 * (jl_max - jl_min) + jl_min


def load_variants(fname):
    """Return dict with the three jl arrays + shared metadata for one session."""
    stem = os.path.splitext(fname)[0]
    with open(os.path.join(WEB_DATA, fname)) as f:
        d = json.load(f)

    T, L, D = d["n_tokens"], d["n_layers"], d["jl_dim"]

    # (a) published pathway: base64 uint8 -> dequantize
    q = np.frombuffer(base64.b64decode(d["jl"]), dtype=np.uint8).reshape(T, L, D)
    jl_min_json = np.array(d["jl_min"], dtype=np.float32)
    jl_max_json = np.array(d["jl_max"], dtype=np.float32)
    jl_quant = dequantize(q, jl_min_json, jl_max_json)

    # (b) ground truth float
    npz = np.load(os.path.join(NPZ_DATA, f"{stem}_activations.npz"))
    jl_float = npz["jl"].astype(np.float32)
    assert jl_float.shape == (T, L, D), f"{stem}: npz {jl_float.shape} != json {(T, L, D)}"

    # (c) software round-trip of (b)
    q2, mn2, mx2 = quantize_jl(jl_float)
    jl_rtrip = dequantize(q2, mn2, mx2)

    return {
        "stem": stem,
        "display_name": d.get("display_name", stem),
        "n_tokens": T,
        "n_layers": L,
        "token_pieces": d["token_pieces"],
        "labels": [token_affect_label(tp) for tp in d["token_pieces"]],
        "QUANT": jl_quant,
        "FLOAT": jl_float,
        "RTRIP": jl_rtrip,
        "q_bytes_a": q,
        "q_bytes_c": q2,
        "jl_min": mn2,
        "jl_max": mx2,
        "spread": (mx2 - mn2),
    }


def concat80(jl, n_layers):
    """80D concatenation over LAYERS_OF_INTEREST, as run_analysis.py step 3."""
    assert all(L < n_layers for L in LAYERS_OF_INTEREST)
    return np.concatenate([jl[:, L, :] for L in LAYERS_OF_INTEREST], axis=1)


def build_direction(sessions, variant):
    """Pooled 80D affect direction across all sessions (run_analysis.py step 2)."""
    pos, neg = [], []
    for s in sessions:
        v80 = concat80(s[variant], s["n_layers"])
        for i, lbl in enumerate(s["labels"]):
            if lbl == "positive":
                pos.append(v80[i])
            elif lbl == "negative":
                neg.append(v80[i])
    pos_mean = np.mean(np.asarray(pos, dtype=np.float64), axis=0)
    neg_mean = np.mean(np.asarray(neg, dtype=np.float64), axis=0)
    direction = pos_mean - neg_mean
    n = np.linalg.norm(direction)
    if n > 0:
        direction = direction / n
    return direction, len(pos), len(neg)


def baselines(sessions, variant, direction):
    """Tonic baseline per session = mean 80D projection of label-None tokens."""
    out = {}
    for s in sessions:
        v80 = concat80(s[variant], s["n_layers"]).astype(np.float64)
        proj = v80 @ direction
        mask = np.array([lbl is None for lbl in s["labels"]])
        out[s["stem"]] = float(np.mean(proj[mask])) if mask.sum() else float(np.mean(proj))
    return out


def per_layer_direction_cos(sessions):
    """Cosine similarity of the 16D per-layer directions, FLOAT vs QUANT."""
    rows = []
    for L in LAYERS_OF_INTEREST:
        d = {}
        for variant in ("FLOAT", "QUANT"):
            pos, neg = [], []
            for s in sessions:
                jl = s[variant]
                for i, lbl in enumerate(s["labels"]):
                    if lbl == "positive":
                        pos.append(jl[i, L, :])
                    elif lbl == "negative":
                        neg.append(jl[i, L, :])
            v = np.mean(np.asarray(pos, np.float64), 0) - np.mean(np.asarray(neg, np.float64), 0)
            d[variant] = v / np.linalg.norm(v)
        rows.append((L, float(d["FLOAT"] @ d["QUANT"])))
    return rows


def main():
    print("=" * 100)
    print("AFFECT / TONIC-BASELINE QUANTIZATION RECHECK")
    print("=" * 100)

    sessions = [load_variants(f) for f in SESSION_FILES]

    # sanity: does (c) reproduce (a) exactly?  If yes the shipped JSON is a
    # faithful uint8 encoding of the npz float array and nothing else changed.
    print("\n[0] Encoder reproducibility: software round-trip vs shipped bytes")
    for s in sessions:
        same = np.array_equal(s["q_bytes_a"], s["q_bytes_c"])
        nd = int((s["q_bytes_a"] != s["q_bytes_c"]).sum())
        print(f"    {s['stem']:26s} bytes identical: {str(same):5s} (differing: {nd})")

    # ---------------- directions ----------------
    print("\n[1] Pooled 80D affect direction (layers 11/22/33/44/55)")
    dirs, counts = {}, {}
    for v in ("QUANT", "FLOAT", "RTRIP"):
        dirs[v], npos, nneg = build_direction(sessions, v)
        counts[v] = (npos, nneg)
        print(f"    {v:6s}: {npos} positive tokens, {nneg} negative tokens, "
              f"||d||={np.linalg.norm(dirs[v]):.6f}")
    print(f"    cos(FLOAT, QUANT) = {dirs['FLOAT'] @ dirs['QUANT']:.6f}")
    print(f"    cos(FLOAT, RTRIP) = {dirs['FLOAT'] @ dirs['RTRIP']:.6f}")
    print(f"    cos(QUANT, RTRIP) = {dirs['QUANT'] @ dirs['RTRIP']:.6f}")

    # ---------------- baselines ----------------
    B = {v: baselines(sessions, v, dirs[v]) for v in ("QUANT", "FLOAT", "RTRIP")}

    published = None
    pub_path = os.path.join(REPO, "analysis_results.json")
    if os.path.exists(pub_path):
        with open(pub_path) as f:
            published = json.load(f)["per_session_baseline"]

    print("\n[2] TONIC BASELINE, three ways")
    hdr = (f"{'session':26s} {'PUBLISHED':>11s} {'(a) QUANT':>11s} {'(b) FLOAT':>11s} "
           f"{'(c) RTRIP':>11s} {'(a)-(b)':>10s} {'sign a':>7s} {'sign b':>7s}")
    print("    " + hdr)
    print("    " + "-" * len(hdr))
    for s in sessions:
        k = s["stem"]
        p = published.get(k, float("nan")) if published else float("nan")
        a, b, c = B["QUANT"][k], B["FLOAT"][k], B["RTRIP"][k]
        print(f"    {k:26s} {p:11.4f} {a:11.4f} {b:11.4f} {c:11.4f} {a-b:10.4f} "
              f"{('-' if a<0 else '+'):>7s} {('-' if b<0 else '+'):>7s}")

    for v in ("QUANT", "FLOAT", "RTRIP"):
        vals = np.array([B[v][s["stem"]] for s in sessions])
        print(f"\n    {v:6s}: grand mean {vals.mean():+.4f} | "
              f"n_negative {int((vals<0).sum())}/8 | "
              f"range [{vals.min():+.4f}, {vals.max():+.4f}] | "
              f"sd {vals.std(ddof=1):.4f}")

    # ---------------- effect-size context ----------------
    print("\n[3] (a)-(b) shift relative to effect size")
    fvals = np.array([B["FLOAT"][s["stem"]] for s in sessions])
    qvals = np.array([B["QUANT"][s["stem"]] for s in sessions])
    shift = qvals - fvals
    print(f"    mean shift            : {shift.mean():+.4f}")
    print(f"    between-session sd of FLOAT baselines : {fvals.std(ddof=1):.4f}")
    print(f"    |mean shift| / sd(FLOAT)              : {abs(shift.mean())/fvals.std(ddof=1):.3f}")
    print(f"    |mean shift| / |mean FLOAT baseline|  : "
          f"{abs(shift.mean())/abs(fvals.mean()):.3f}" if fvals.mean() != 0 else "")

    # per-session, projections are also compared against the within-session
    # spread of neutral-token projections (the natural noise scale)
    print(f"\n    {'session':26s} {'shift':>10s} {'sd(neutral proj, FLOAT)':>24s} {'shift/sd':>10s}")
    for s in sessions:
        v80 = concat80(s["FLOAT"], s["n_layers"]).astype(np.float64)
        proj = v80 @ dirs["FLOAT"]
        mask = np.array([l is None for l in s["labels"]])
        sd = proj[mask].std(ddof=1)
        sh = B["QUANT"][s["stem"]] - B["FLOAT"][s["stem"]]
        print(f"    {s['stem']:26s} {sh:10.4f} {sd:24.4f} {sh/sd:10.4f}")

    # ---------------- truncation bias ----------------
    print("\n[4] Truncation bias: mean(dequantized - original), per dimension")
    print("    (over all T*L entries; theory says -0.5*spread/255)")
    for s in sessions:
        err = (s["RTRIP"].astype(np.float64) - s["FLOAT"].astype(np.float64))
        bias_per_dim = err.reshape(-1, 16).mean(axis=0)
        theory = -0.5 * s["spread"].astype(np.float64) / 255.0
        print(f"\n    {s['stem']}")
        print(f"      per-dim spread      min={s['spread'].min():9.2f} max={s['spread'].max():9.2f}")
        print(f"      1 LSB (spread/255)  min={s['spread'].min()/255:9.4f} max={s['spread'].max()/255:9.4f}")
        print(f"      observed bias/dim   min={bias_per_dim.min():9.4f} max={bias_per_dim.max():9.4f} "
              f"mean={bias_per_dim.mean():9.4f}")
        print(f"      theoretical -0.5LSB min={theory.min():9.4f} max={theory.max():9.4f} "
              f"mean={theory.mean():9.4f}")
        print(f"      all dims negative?  {bool((bias_per_dim < 0).all())}")

        # contribution of that bias to the 80D projection
        bias80 = []
        for L in LAYERS_OF_INTEREST:
            e = err[:, L, :].mean(axis=0)
            bias80.append(e)
        bias80 = np.concatenate(bias80)
        contrib = float(bias80 @ dirs["FLOAT"])
        actual = B["QUANT"][s["stem"]] - B["FLOAT"][s["stem"]]
        print(f"      bias80 . d_FLOAT    = {contrib:+.4f}   (actual (a)-(b) = {actual:+.4f})")

    # ---------------- per-layer direction stability ----------------
    print("\n[5] Direction stability, FLOAT vs QUANT (16D per-layer directions)")
    for L, cos in per_layer_direction_cos(sessions):
        print(f"    layer {L:2d}: cos = {cos:.6f}")

    # ---------------- rounding counterfactual ----------------
    print("\n[6] Counterfactual: same encoder but with rounding instead of truncation")
    round_jl = {}
    for s in sessions:
        jl = s["FLOAT"]
        mn, mx = s["jl_min"], s["jl_max"]
        spread = (mx - mn).copy()
        spread[spread < 1e-10] = 1.0
        q = np.rint(np.clip((jl - mn) / spread, 0, 1) * 255).astype(np.uint8)
        round_jl[s["stem"]] = dequantize(q, mn, mx)
    for s in sessions:
        s["ROUND"] = round_jl[s["stem"]]
    dR, _, _ = build_direction(sessions, "ROUND")
    BR = baselines(sessions, "ROUND", dR)
    rv = np.array([BR[s["stem"]] for s in sessions])
    print(f"    {'session':26s} {'ROUND':>11s} {'FLOAT':>11s} {'ROUND-FLOAT':>12s}")
    for s in sessions:
        print(f"    {s['stem']:26s} {BR[s['stem']]:11.4f} {B['FLOAT'][s['stem']]:11.4f} "
              f"{BR[s['stem']]-B['FLOAT'][s['stem']]:12.4f}")
    print(f"    grand mean {rv.mean():+.4f} | n_negative {int((rv<0).sum())}/8")

    # ---------------- verdict ----------------
    print("\n" + "=" * 100)
    fn = int((fvals < 0).sum())
    qn = int((qvals < 0).sum())
    print(f"VERDICT: '8/8 negative' -- QUANT: {qn}/8 negative | FLOAT: {fn}/8 negative")
    flips = [s["stem"] for s in sessions
             if (B["QUANT"][s["stem"]] < 0) != (B["FLOAT"][s["stem"]] < 0)]
    print(f"Sessions whose sign differs between (a) and (b): {flips if flips else 'none'}")
    print("=" * 100)

    out = {
        "quant": B["QUANT"], "float": B["FLOAT"], "rtrip": B["RTRIP"], "round": BR,
        "published": published,
        "cos_float_quant_80d": float(dirs["FLOAT"] @ dirs["QUANT"]),
        "n_pos_neg": {k: list(v) for k, v in counts.items()},
    }
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "affect_float_recheck_results.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
