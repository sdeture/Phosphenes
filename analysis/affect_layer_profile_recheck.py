#!/usr/bin/env python3
"""
affect_layer_profile_recheck.py -- companion to affect_float_recheck.py.

Two extra things the headline table does not cover:

  (A) baseline_investigation.py's run_layer0_control() layer profile (the piece
      the published Correction note still calls "a valid structural
      observation"), recomputed on float vs quantized data. Same pooling as the
      original: one 16D direction per layer, pooled over all 8 sessions, and a
      single pooled baseline over all neutral tokens.

  (B) Whether the per-session 80D baselines are distinguishable from zero at
      all, under float. Reported as a naive one-sample t over neutral tokens
      (optimistic: tokens within a session are not independent) and as a
      session-level one-sample t over the 8 session means.

  (C) Decomposition of the (a)-(b) gap into "shifted data" vs "rotated
      direction" contributions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from affect_float_recheck import (  # noqa: E402
    LAYERS_OF_INTEREST, SESSION_FILES, baselines, build_direction, concat80,
    load_variants,
)

TEST_LAYERS = [0, 5, 11, 16, 22, 27, 33, 38, 44, 49, 55, 60, 63]


def layer_profile(sessions, variant):
    """Reproduce run_layer0_control(): pooled 16D direction + pooled baseline."""
    out = {}
    for L in TEST_LAYERS:
        pos, neg = [], []
        for s in sessions:
            jl = s[variant]
            for i, lbl in enumerate(s["labels"]):
                if lbl == "positive":
                    pos.append(jl[i, L, :])
                elif lbl == "negative":
                    neg.append(jl[i, L, :])
        pm = np.mean(np.asarray(pos, np.float64), 0)
        nm = np.mean(np.asarray(neg, np.float64), 0)
        direction = pm - nm
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 0 else direction

        projs, labs = [], []
        for s in sessions:
            projs.append(s[variant][:, L, :].astype(np.float64) @ direction)
            labs.extend(s["labels"])
        projs = np.concatenate(projs)
        labs = np.array([l is None for l in labs])
        base = float(projs[labs].mean())

        pv = projs[np.array([l == "positive" for s in sessions for l in s["labels"]])]
        nv = projs[np.array([l == "negative" for s in sessions for l in s["labels"]])]
        ps = np.sqrt((pv.var(ddof=1) + nv.var(ddof=1)) / 2)
        d = float((pv.mean() - nv.mean()) / ps) if ps > 0 else 0.0

        out[L] = {"baseline": base, "cohens_d": d, "dir_mag": norm,
                  "direction": direction}
    return out


def main():
    sessions = [load_variants(f) for f in SESSION_FILES]

    print("=" * 96)
    print("(A) LAYER PROFILE (baseline_investigation.py run_layer0_control), float vs quantized")
    print("=" * 96)
    pf = layer_profile(sessions, "FLOAT")
    pq = layer_profile(sessions, "QUANT")

    # per-dim truncation bias vector, averaged over sessions weighted by tokens
    print(f"\n{'layer':>5} {'QUANT base':>12} {'FLOAT base':>12} {'diff':>10} "
          f"{'sign Q':>7} {'sign F':>7} {'cos(dF,dQ)':>11} {'d QUANT':>8} {'d FLOAT':>8}")
    print("-" * 96)
    flips = []
    for L in TEST_LAYERS:
        q, f = pq[L]["baseline"], pf[L]["baseline"]
        cos = float(pf[L]["direction"] @ pq[L]["direction"])
        if (q < 0) != (f < 0):
            flips.append(L)
        print(f"{L:5d} {q:12.4f} {f:12.4f} {q-f:10.4f} "
              f"{('-' if q<0 else '+'):>7} {('-' if f<0 else '+'):>7} {cos:11.6f} "
              f"{pq[L]['cohens_d']:8.4f} {pf[L]['cohens_d']:8.4f}")
    print(f"\n  layers whose baseline SIGN differs float vs quantized: {flips}")
    nq = sum(1 for L in TEST_LAYERS if pq[L]["baseline"] < 0)
    nf = sum(1 for L in TEST_LAYERS if pf[L]["baseline"] < 0)
    print(f"  negative layers: QUANT {nq}/{len(TEST_LAYERS)}  FLOAT {nf}/{len(TEST_LAYERS)}")

    # true scale of the residual stream per layer, vs the quantization step
    print("\n  Per-layer signal scale vs the (global) quantization step:")
    print(f"  {'layer':>5} {'RMS |jl| (float)':>18} {'mean 1 LSB':>12} {'LSB/RMS':>9}")
    lsb = np.mean([s["spread"] / 255.0 for s in sessions], axis=0).mean()
    for L in TEST_LAYERS:
        rms = float(np.sqrt(np.mean([np.mean(s["FLOAT"][:, L, :].astype(np.float64) ** 2)
                                     for s in sessions])))
        print(f"  {L:5d} {rms:18.3f} {lsb:12.4f} {lsb/rms:9.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("(B) Is the 80D per-session baseline distinguishable from zero on FLOAT data?")
    print("=" * 96)
    dF, _, _ = build_direction(sessions, "FLOAT")
    dQ, _, _ = build_direction(sessions, "QUANT")
    BF = baselines(sessions, "FLOAT", dF)
    BQ = baselines(sessions, "QUANT", dQ)

    print(f"\n{'session':26s} {'FLOAT base':>11} {'n_neutral':>10} {'SEM':>9} "
          f"{'t':>8} {'QUANT base':>11} {'t (quant)':>10}")
    print("-" * 96)
    for s in sessions:
        v80 = concat80(s["FLOAT"], s["n_layers"]).astype(np.float64)
        proj = v80 @ dF
        m = np.array([l is None for l in s["labels"]])
        x = proj[m]
        sem = x.std(ddof=1) / np.sqrt(x.size)
        v80q = concat80(s["QUANT"], s["n_layers"]).astype(np.float64)
        pq_ = v80q @ dQ
        xq = pq_[m]
        semq = xq.std(ddof=1) / np.sqrt(xq.size)
        print(f"{s['stem']:26s} {BF[s['stem']]:11.4f} {x.size:10d} {sem:9.4f} "
              f"{BF[s['stem']]/sem:8.3f} {BQ[s['stem']]:11.4f} {BQ[s['stem']]/semq:10.3f}")

    fv = np.array([BF[s["stem"]] for s in sessions])
    qv = np.array([BQ[s["stem"]] for s in sessions])
    for nm, v in (("FLOAT", fv), ("QUANT", qv)):
        sem = v.std(ddof=1) / np.sqrt(v.size)
        print(f"\n  session-level one-sample t ({nm}): mean {v.mean():+.4f}, "
              f"sem {sem:.4f}, t(7) = {v.mean()/sem:+.3f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("(C) Decomposition of the (a)-(b) gap: shifted data vs rotated direction")
    print("=" * 96)
    B_fdata_qdir = baselines(sessions, "FLOAT", dQ)
    B_qdata_fdir = baselines(sessions, "QUANT", dF)
    print(f"\n{'session':26s} {'(b) F/dF':>10} {'F data,dQ':>10} {'Q data,dF':>10} "
          f"{'(a) Q/dQ':>10} {'data effect':>12} {'dir effect':>11}")
    print("-" * 96)
    for s in sessions:
        k = s["stem"]
        print(f"{k:26s} {BF[k]:10.4f} {B_fdata_qdir[k]:10.4f} {B_qdata_fdir[k]:10.4f} "
              f"{BQ[k]:10.4f} {B_qdata_fdir[k]-BF[k]:12.4f} {B_fdata_qdir[k]-BF[k]:11.4f}")
    print(f"\n  cos(dF,dQ) = {float(dF @ dQ):.6f}")
    print("  'data effect'  = same direction, quantized vs float data (the truncation bias)")
    print("  'dir effect'   = same float data, direction re-estimated on quantized data")


if __name__ == "__main__":
    main()
