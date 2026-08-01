#!/usr/bin/env python3
"""
verify_tour_claims.py — Check every number the guided tour asserts.

The tour tells a viewer to trust their eyes. That only works if the things it
points at are really there, so each numeric claim in `web/js/tours.js` is
restated here as an assertion with a tolerance, and this script recomputes it
from source. Exit status is non-zero if any claim has drifted.

Run it after changing the data pipeline, the seam definition, or the tour copy:

    python analysis/verify_tour_claims.py

Two sources are used deliberately:

  * `data/*_activations.npz` — float32 ground truth. Claims about the MODEL are
    checked here (layer energy profile, entropy profile, prefix identity).
  * `web/data/*.json` — the quantised bundles the viewer actually displays.
    Claims about what a viewer will SEE are checked here (seam scores). A claim
    verified only against float data could still be invisible in the shipped
    build, which is the failure mode this split is meant to catch.
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEB = ROOT / "web" / "data"

sys.path.insert(0, str(ROOT))
import activations  # noqa: E402  (needs ROOT on the path first)

FLAGSHIP = "Dream_greedy_clean"
FORK_OTHER = "Dream_greedy_sentient"
ALL_STEMS = [
    "Dream_greedy_clean", "Dream_greedy_sentient",
    "Dream_conv_00173_run1", "Dream_conv_00178_run1", "Dream_conv_00181_run1",
    "Dream_conv_00187_run1", "Dream_conv_00191_run1", "Dream_conv_00194_run1",
]

IM_START, IM_END = 151644, 151645

results = []


def check(label, got, expected, tol, unit=""):
    """Record a numeric claim. `tol` is absolute."""
    ok = abs(got - expected) <= tol
    results.append((ok, label, f"{got:.4g}{unit}", f"{expected:.4g}{unit} ±{tol:g}"))
    return ok


def check_bool(label, got, expected=True):
    results.append((got == expected, label, str(got), str(expected)))
    return got == expected


def npz(stem):
    return np.load(str(DATA / f"{stem}_activations.npz"))


def bundle(stem):
    return json.loads((WEB / f"{stem}.json").read_text())


def u8(d, key):
    return np.frombuffer(base64.b64decode(d[key]), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — "brightness rises as you look upward"
# ══════════════════════════════════════════════════════════════════════

a = npz(FLAGSHIP)
energy = a["jl_energy"].astype(np.float32).mean(axis=0)
check("step1: mean JL magnitude at layer 0", float(energy[0]), 17.8, 0.5)
check("step1: mean JL magnitude at layer 60", float(energy[60]), 1381.7, 5.0)
check("step1: growth factor layer 0 -> 60 (sketch)", float(energy[60] / energy[0]), 78.0, 2.0, "x")

# The same quantity measured exactly, which is what the README now quotes. The
# sketch and the exact norm disagree by ~18% on the growth factor; both numbers
# are asserted so that neither can be quoted as the other by accident.
hn = a["h_norm"].astype(np.float32)
h_mean = hn.mean(axis=0)
check("step1: mean EXACT norm at layer 0", float(h_mean[0]), 19.9, 0.5)
check("step1: mean EXACT norm at layer 60", float(h_mean[60]), 1308.8, 5.0)
check("step1: growth factor layer 0 -> 60 (exact)",
      float(h_mean[60] / h_mean[0]), 65.6, 2.0, "x")
check_bool("step1: sketch overstates the growth, and README quotes the exact one",
           bool(energy[60] / energy[0] > h_mean[60] / h_mean[0]))

# ...and the reason, asserted, because it is the part a reader will not believe:
# the sketch/exact ratio DRIFTS with depth. Sampling noise would not do this —
# the s.e.m. of each layer's ratio over ~3,000 tokens is about 0.003.
je = a["jl_energy"].astype(np.float32)
ratio = (je / hn).mean(axis=0)
check("step1: sketch/exact ratio at layer 0", float(ratio[0]), 0.887, 0.02)
check("step1: sketch/exact ratio at layer 60", float(ratio[60]), 1.052, 0.02)
check("step1: ratio drift explains the growth discrepancy",
      float((ratio[60] / ratio[0]) / ((energy[60] / energy[0]) / (h_mean[60] / h_mean[0]))),
      1.0, 0.01, "x")
check_bool("step1: per-cell sketch error matches JL theory for k=16",
           bool(0.10 < float((je / hn).std()) < 0.25))

# What the VIEWER actually shows, which is not the same thing. energy_norm is
# quantile-normalised per layer (convert_for_web.py), so the 66x growth above is
# emphatically not on screen; the step-1 caption quotes the on-screen numbers
# separately for exactly that reason. Read from the shipped bundle on purpose:
# this is a claim about the display format, not about the model.
bright = u8(bundle(FLAGSHIP), "energy_norm").astype(np.float32)
bright = bright.reshape(-1, 64).mean(axis=0)
check("step1: mean DISPLAY brightness at layer 0", float(bright[0]), 103.0, 2.0, "/255")
check("step1: mean DISPLAY brightness at layer 60", float(bright[60]), 151.0, 2.0, "/255")

# ══════════════════════════════════════════════════════════════════════
# SEAMS — they fire at turn ENDINGS, not beginnings. Quoted in step 5's
# evidence; the standalone seam step was cut from the tour on 2026-07-31, but
# seams are still drawn (legend, scrubber ticks, bookmarks) so the measurement
# still has to hold.
# Checked on the shipped bundles: this is a claim about what is visible.
# ══════════════════════════════════════════════════════════════════════

end_means, start_means, other_means = [], [], []
for stem in ALL_STEMS:
    d = bundle(stem)
    seam = u8(d, "seam_score").astype(np.float32) / 255.0
    ids = np.load(str(DATA / f"{stem}_input_ids.npy"))
    ends = [i for i, t in enumerate(ids) if int(t) == IM_END]
    starts = [i for i, t in enumerate(ids) if int(t) == IM_START and i > 0]
    mask = np.ones(len(seam), bool)
    mask[0] = False                     # excluded by construction
    for i in ends + starts:
        mask[i] = False
    end_means.append(seam[ends].mean())
    start_means.append(seam[starts].mean())
    other_means.append(seam[mask].mean())

end_m = float(np.mean(end_means))
start_m = float(np.mean(start_means))
other_m = float(np.mean(other_means))
ratios = np.array(end_means) / np.array(other_means)

check("seam: mean seam at <|im_end|>", end_m, 0.780, 0.02)
check("seam: mean seam elsewhere", other_m, 0.110, 0.02)
check("seam: mean seam at <|im_start|>", start_m, 0.004, 0.02)
check("seam: im_end / elsewhere ratio", float(ratios.mean()), 7.1, 0.8, "x")
check("seam: ratio range low", float(ratios.min()), 6.2, 0.4, "x")
check("seam: ratio range high", float(ratios.max()), 7.8, 0.9, "x")
check_bool("seam: im_end > im_start in all 8 sessions",
           bool(np.all(np.array(end_means) > np.array(start_means))))

flag = bundle(FLAGSHIP)
seam_flag = u8(flag, "seam_score").astype(np.float32) / 255.0
check_bool("seam: token 0 seam suppressed to zero", float(seam_flag[0]) == 0.0)

# The 60th-percentile floor makes this a detector, not a continuous measure. The
# docs said "roughly 40% of tokens are pinned to zero" until 2026-07-28 — the
# percentile read as the survivor fraction rather than the floor.
zero_frac = float(np.mean([(u8(bundle(s), "seam_score") == 0).mean() for s in ALL_STEMS]))
check("seam: fraction of tokens pinned to zero", zero_frac, 0.601, 0.02)

# The one quantitative claim that lived in shipping code (config.js) with no
# assertion behind it. It is also window-sensitive, so the window is pinned here.
ids_f = np.load(str(DATA / f"{FLAGSHIP}_input_ids.npy"))
bounds = [i for i, t in enumerate(ids_f) if int(t) in (IM_START, IM_END)]
near = np.zeros(len(seam_flag), bool)
for i in bounds:
    near[max(0, i - 2):i + 3] = True
near[0] = False
far = ~near
far[0] = False
check("seam: seam ratio within +/-2 tokens of a turn boundary",
      float(seam_flag[near].mean() / seam_flag[far].mean()), 2.85, 0.15, "x")
check("seam: seam at token 193 (im_end)", float(seam_flag[193]), 0.972, 0.02)

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — two sparsity bands, maxima near layers 10 and 40
# ══════════════════════════════════════════════════════════════════════

sp = (a["top1_frac"].astype(np.float32) * 0.6
      + a["top25_frac"].astype(np.float32) * 0.4).mean(axis=0)
lo_band = int(np.argmax(sp[:24]))
hi_band = 24 + int(np.argmax(sp[24:52]))
trough = 24 + int(np.argmin(sp[24:36]))
check("step3: lower sparsity maximum layer", lo_band, 9, 1)
check("step3: lower maximum value", float(sp[lo_band]), 0.548, 0.01)
check("step3: upper sparsity maximum layer", hi_band, 44, 2)
check("step3: upper maximum value", float(sp[hi_band]), 0.575, 0.02)
check("step3: trough layer", trough, 28, 4)
check("step3: trough value", float(sp[trough]), 0.463, 0.02)

# Banding must be a property of the network, not of one conversation.
both_bands = 0
for stem in ALL_STEMS:
    s = npz(stem)
    p = (s["top1_frac"].astype(np.float32) * 0.6
         + s["top25_frac"].astype(np.float32) * 0.4).mean(axis=0)
    lo_i = int(np.argmax(p[:24]))
    hi_i = 24 + int(np.argmax(p[24:52]))
    tr_i = 24 + int(np.argmin(p[24:36]))
    if p[lo_i] > p[tr_i] and p[hi_i] > p[tr_i]:
        both_bands += 1
check("step3: sessions showing both bands", both_bands, 8, 0)

# `top1_frac` is the top one PERCENT of dimensions (k = 52 of 5,120), not the
# single largest. Documented as max(s)/sum(s) until 2026-07-28. The means below
# are the doc's own quoted figures; the bound is the thing that makes the
# distinction checkable, since one dimension out of 5,120 carrying a mean 0.29
# of all token-to-token change would be an extraordinary claim.
t1 = a["top1_frac"].astype(np.float32)[1:]      # row 0 undefined, see below
t25 = a["top25_frac"].astype(np.float32)[1:]
check("step3: top1_frac mean (top 1% of dims)", float(t1.mean()), 0.29, 0.02)
check("step3: top25_frac mean (top 25% of dims)", float(t25.mean()), 0.83, 0.02)
check_bool("step3: top1_frac nests inside top25_frac", bool((t1 <= t25 + 1e-6).all()))

# Token 0 has no predecessor, so every difference-derived display array must be
# zeroed there. Only seam_score was, until 2026-07-28 — cos_instability shipped
# saturated at 255, rendering the first column of every session at full grain.
for stem in ALL_STEMS:
    d = bundle(stem)
    L0 = d["n_layers"]
    for key in ("delta_norm", "cos_instability", "sparsity_norm"):
        check_bool(f"token0: {key} zeroed at token 0 ({stem[:16]})",
                   bool(np.all(u8(d, key)[:L0] == 0)))

# The float source keeps the raw zeros on purpose — it is the record, not a
# display. Assert they are still there so the repair stays in the display layer.
check_bool("token0: float source still carries the raw undefined values",
           bool(np.all(a["cos_prev"].astype(np.float32)[0] == 0.0)))

# Token 0's large magnitude is NOT that artefact — it needs no predecessor. This
# is the attention-sink / massive-activation effect, and it is real.
check_bool("token0: first-token norm is genuinely large (attention sink)",
           bool(hn[0].mean() / hn[1:].mean() > 10.0))

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — logit-lens entropy rises then falls
#
# Read through activations.logit_lens_entropy(), which repairs the top layer.
# The extracted top layer is double-normalised; the block below asserts both
# that the repair works and that the defect it repairs is still there, so the
# substitution cannot rot into a no-op without failing this script.
# ══════════════════════════════════════════════════════════════════════

ent = activations.logit_lens_entropy(a).mean(axis=0)
peak = int(np.argmax(ent[:24]))
check("step4: entropy at layer 0", float(ent[0]), 8.84, 0.1, " nats")
check("step4: entropy peak layer", peak, 9, 1)
check("step4: entropy at peak", float(ent[peak]), 9.84, 0.05, " nats")
check("step4: entropy at final layer", float(ent[-1]), 1.00, 0.05, " nats")
check("step4: ln(vocab) reference", float(np.log(151936)), 11.93, 0.01, " nats")
check_bool("step4: entropy rises before it falls", bool(ent[peak] > ent[0] and ent[-1] < ent[0]))

rise_in_all = sum(
    1 for stem in ALL_STEMS
    if (lambda e: e[int(np.argmax(e[:24]))] > e[0] and e[-1] < e[0])(
        activations.logit_lens_entropy(npz(stem)).mean(axis=0))
)
check("step4: sessions showing the rise-then-fall", rise_in_all, 8, 0)

# The repair, both directions, in every session.
repaired_ok, defect_present = 0, 0
for stem in ALL_STEMS:
    s = npz(stem)
    fixed = activations.logit_lens_entropy(s)[:, -1]
    raw = activations.raw_logit_lens_entropy(s)[:, -1]
    truth = s["token_entropy"].astype(np.float32)
    if float(np.abs(fixed - truth).max()) == 0.0:
        repaired_ok += 1
    if float(np.abs(raw - truth).max()) > 0.5:
        defect_present += 1
check("step4: repaired top layer == true output entropy, all sessions",
      repaired_ok, 8, 0)
check("step4: raw top layer still shows the extraction defect",
      defect_present, 8, 0)

# `logit_lens_rank` carries the identical defect and the identical repair. It is
# unused today, which is why it needs an assertion: METRICS §6 advertises it as
# the cheapest overlay to add, so the next person to wire it up inherits this.
rank_ok, rank_defect = 0, 0
for stem in ALL_STEMS:
    s = npz(stem)
    if np.array_equal(activations.logit_lens_rank(s)[:, -1],
                      s["actual_rank"].astype(np.int32)):
        rank_ok += 1
    if np.mean(s["logit_lens_rank"].astype(np.int32)[:, -1]
               == s["actual_rank"].astype(np.int32)) < 0.9:
        rank_defect += 1
check("step4: repaired logit_lens_rank top layer == actual_rank", rank_ok, 8, 0)
check("step4: raw logit_lens_rank still shows the same defect", rank_defect, 8, 0)

# The layer beneath is untouched by the defect and must stay above the output:
# if this inverts, the repair has been applied to the wrong column.
check_bool("step4: layer 62 entropy exceeds the output entropy",
           bool(ent[62] > ent[-1]))

# The claim the correction strengthens rather than weakens: the true output is
# LOWER than the withdrawn 1.65 figure, so the model commits harder, not less.
check_bool("step4: corrected output entropy is below the withdrawn 1.65",
           bool(ent[-1] < 1.65))

# ══════════════════════════════════════════════════════════════════════
# THE SESSION SET — the colour basis and the normalisation bounds must be
# built from the same sessions. They were not: compute_shared_pca.py globbed
# all nine while convert_for_web.py pooled eight, so every shipped colour
# depended on a session declared held out of both viewers.
# ══════════════════════════════════════════════════════════════════════

pca_t = np.load(str(DATA / "shared_pca_transform.npz"))
check_bool("sessions: shared PCA records the stems it was fitted on",
           "stems" in pca_t.files)
fitted = sorted(str(s) for s in pca_t["stems"]) if "stems" in pca_t.files else []
check("sessions: PCA fitted on the displayed session count", len(fitted), 8, 0)
check_bool("sessions: PCA session set == the shipped session set",
           fitted == sorted(ALL_STEMS))
check_bool("sessions: the held-out stem is absent from the colour basis",
           "Dream_greedy_baseline" not in fitted)
check_bool("sessions: the held-out stem is still on disk, not deleted",
           (DATA / "Dream_greedy_baseline_activations.npz").exists())
check_bool("sessions: the held-out stem ships no web bundle",
           not (WEB / "Dream_greedy_baseline.json").exists())

# The colour carries 73% of the sketch's variance, and that is worth asserting
# because it is the number a reader would want and it appears nowhere on screen.
check("sessions: PCA explained variance, 3 components",
      float(pca_t["explained_variance_ratio"].sum()), 0.733, 0.02)

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — turn boundaries of the flagship session
# ══════════════════════════════════════════════════════════════════════

turns = {t["turn"]: t for t in flag["turns"]}
check("step5: turn 5 (user) starts", turns[5]["token_start"], 1310, 0)
check("step5: turn 5 ends (exclusive)", turns[5]["token_end"], 2432, 0)
check("step5: turn 5 length in tokens", turns[5]["token_end"] - turns[5]["token_start"], 1122, 0)
check("step5: turn 6 (assistant) starts", turns[6]["token_start"], 2433, 0)
check("step5: seam at closing token 2431", float(seam_flag[2431]), 1.0, 0.06)
check("step5: seam at opening token 2433", float(seam_flag[2433]), 0.0, 0.06)

# ══════════════════════════════════════════════════════════════════════
# STEP 8 — the one-token fork
# ══════════════════════════════════════════════════════════════════════

ids_l = np.load(str(DATA / f"{FLAGSHIP}_input_ids.npy"))
ids_r = np.load(str(DATA / f"{FORK_OTHER}_input_ids.npy"))
n = min(len(ids_l), len(ids_r))
first_diff = int(np.argmax(ids_l[:n] != ids_r[:n]))
check("fork: first differing token index", first_diff, 73, 0)
check_bool("fork: prefix token ids identical", bool(np.array_equal(ids_l[:73], ids_r[:73])))

jl_l = npz(FLAGSHIP)["jl"].astype(np.float32)
jl_r = npz(FORK_OTHER)["jl"].astype(np.float32)
check("fork: max |difference| over prefix (float source)",
      float(np.abs(jl_l[:73] - jl_r[:73]).max()), 0.0, 0.0)

dist = np.linalg.norm(jl_l[:n] - jl_r[:n], axis=2)      # (n, L)
per_pos = dist.mean(axis=1)
check("fork: mean distance at token 73", float(per_pos[73]), 280.5, 6.0)
check("fork: mean distance, tokens 1000-2900", float(per_pos[1000:2900].mean()), 264.4, 6.0)
check("fork: typical JL magnitude (corpus-wide)",
      float(np.linalg.norm(jl_l, axis=2).mean()), 299.2, 4.0)

# The like-for-like baseline: the magnitude of the state at the SAME token the
# divergence is measured at. Quoted against the corpus-wide mean only, until
# 2026-07-28, which flattered the ratio (0.94 rather than 0.79).
mag_at_fork = float(np.linalg.norm(jl_l[73], axis=1).mean())
check("fork: state magnitude AT token 73", mag_at_fork, 353.0, 5.0)
check("fork: displacement / same-token magnitude", float(per_pos[73]) / mag_at_fork,
      0.79, 0.03, "x")

# ...and the claim that keeps "total separation" from coming back. Orthogonal
# states would give cosine 0 and a ratio of sqrt(2); these give 0.61 and 0.79.
cos_fork = ((jl_l[73] * jl_r[73]).sum(axis=1)
            / (np.linalg.norm(jl_l[73], axis=1) * np.linalg.norm(jl_r[73], axis=1)))
check("fork: mean cosine between the two runs at token 73",
      float(cos_fork.mean()), 0.610, 0.02)
check_bool("fork: the two runs are NOT orthogonal after the fork",
           bool(cos_fork.mean() > 0.4))
check_bool("fork: displacement ratio is well below the orthogonal value sqrt(2)",
           bool(float(per_pos[73]) / mag_at_fork < 1.2))
check("fork: distance at fork, layer 0", float(dist[73, 0]), 36.5, 2.0)
check("fork: distance at fork, layer 62", float(dist[73, 62]), 1286.6, 20.0)
check("fork: left run length", len(ids_l), 2990, 0)
check("fork: right run length", len(ids_r), 3379, 0)

# The property the divergence view rests on: identical inputs must survive the
# pipeline as identical BYTES, or the panes will differ where they must not.
bl, br = bundle(FLAGSHIP), bundle(FORK_OTHER)
for key, per_cell in [("rgb", 3), ("jl", 16), ("energy_norm", 1), ("delta_norm", 1),
                      ("cos_instability", 1), ("sparsity_norm", 1), ("entropy_norm", 1)]:
    L = bl["n_layers"]
    m = 73 * L * per_cell
    check_bool(f"fork: shipped '{key}' identical over prefix",
               bool(np.array_equal(u8(bl, key)[:m], u8(br, key)[:m])))
check_bool("fork: shipped 'seam_score' identical over prefix",
           bool(np.array_equal(u8(bl, "seam_score")[:73], u8(br, "seam_score")[:73])))

# ══════════════════════════════════════════════════════════════════════
# BOOKMARKS — each must still land on a token with the seam it advertises
# ══════════════════════════════════════════════════════════════════════

for tok, expect in [(9, 0.99), (193, 0.972), (239, 0.0), (445, 1.00),
                    (1347, 1.00), (2431, 1.00), (2433, 0.0)]:
    check(f"bookmark: seam at token {tok}", float(seam_flag[tok]), expect, 0.08)

# ══════════════════════════════════════════════════════════════════════
# THE COMPUTE ARGUMENT — every figure quoted in web/about.html, recomputed
# from the recorded architecture.
#
# These exist because the argument was once shipped with a retracted claim in
# it, and none of the assertions above could have caught that: they all check
# the instrument, and the error was in the prose.
# ══════════════════════════════════════════════════════════════════════

meta = json.loads((DATA / f"{FLAGSHIP}_metadata.json").read_text())
d_model, n_layer, n_ctx = meta["d_model"], meta["num_layers"], meta["num_tokens"]
N_PARAMS = 32e9  # from the model id; not derivable from activations

check_bool("compute: model id is a 32B Qwen3-VL", "Qwen3-VL-32B" in meta["model_id"])
check("compute: d_model", d_model, 5120, 0)
check("compute: n_layer", n_layer, 64, 0)
check("compute: n_ctx (flagship)", n_ctx, 2990, 0)

# The convention-free framing: the state the model moves through.
state_numbers = n_ctx * n_layer * d_model
check("compute: state numbers = n_ctx x n_layer x d_model",
      state_numbers, 979_763_200, 0)
check("compute: state size at 2 bytes/value", state_numbers * 2 / 1e9, 1.96, 0.01, " GB")

# Convention (a): marginal decode, cache warm. Kaplan et al. (2020) Table 1.
attn_term = 2 * n_layer * n_ctx * d_model
marginal = 2 * N_PARAMS + attn_term
check("compute: marginal FLOPs/token (2N + attn)", marginal, 6.6e10, 0.1e10)
check("compute: attention share of marginal cost",
      100 * attn_term / marginal, 3.0, 0.5, "%")

# Convention (b): full context, n_ctx x 2N. The four table rows in §1.
for params, ctx, expected in [(25e9, 10_000, 5.0e14), (70e9, 32_000, 4.5e15),
                              (350e9, 70_000, 4.9e16), (400e9, 200_000, 1.6e17)]:
    got = ctx * 2 * params
    check(f"compute: full-context {params/1e9:.0f}B at {ctx//1000}k",
          got, expected, 0.05 * expected)

# The gap between the two conventions IS the context length. That is why the
# docs say "four to six orders of magnitude": the gap is log10(n_ctx), and
# real contexts run 10k-1M. Assert the identity, then the claimed band.
for ctx in (10_000, 200_000, 1_000_000):
    gap = np.log10((ctx * 2 * N_PARAMS) / marginal)
    check(f"compute: convention gap at {ctx//1000}k == log10(n_ctx)",
          float(gap), float(np.log10(ctx)), 0.05, " OOM")
gap_10k = float(np.log10((10_000 * 2 * N_PARAMS) / marginal))
gap_200k = float(np.log10((200_000 * 2 * N_PARAMS) / marginal))
check_bool("compute: '4 to 6 OOM' holds across 10k-200k contexts",
           4.0 <= round(gap_10k, 1) and gap_200k <= 6.0)

# "reads 200k, answers in 10 words" -> ~1.3e15 per output word, this model.
check("compute: 200k read / 10-word answer, per output word",
      (200_000 * 2 * N_PARAMS) / 10, 1.3e15, 0.1e15)

# §3's information rows — the 1:1 correspondence, which is the strongest claim.
check("compute: LLM state accessible/word, generic 8k width at 25k ctx",
      25_000 * 8_000 * 16, 3.2e9, 0.1e9, " bits")
check("compute: LLM state accessible/word, THIS model at 25k ctx",
      25_000 * d_model * 16, 2.0e9, 0.1e9, " bits")
check("compute: new information per token, log2(vocab)",
      float(np.log2(151_936)), 17.2, 0.1, " bits")

# ══════════════════════════════════════════════════════════════════════
# PROSE CONSISTENCY — README.md and web/about.html are the two reader-facing
# surfaces. A correction applied to one and not the other has already happened
# once and shipped; this is the regression test.
#
# docs/THESIS.md and docs/ARCHITECTURE.md were guarded here until 2026-07-31,
# when the documentation was cut to these two files. The retraction guards below
# are the part that had to survive that cut: they are what stops the withdrawn
# framing from reappearing, and they now run against everything that ships.
# ══════════════════════════════════════════════════════════════════════

import re

SURFACES = {
    "web/about.html": (ROOT / "web" / "about.html").read_text(),
    "README.md": (ROOT / "README.md").read_text(),
}

# Retracted claims. If any of these reappears as an assertion, the argument
# has regressed to the version that was withdrawn on 2026-07-27.
RETRACTED = [
    (r"two or three below a brain", "retracted: '2-3 OOM below a brain'"),
    (r"narrow(?:s|ing) the gap", "retracted: gap-direction presupposition"),
    (r"widen(?:s|ing) the gap", "retracted: gap-direction presupposition"),
    (r"10¹³\s*[–-]\s*10¹⁴", "retracted: stale 10^13-10^14 human band"),
    (r"does not survive the numbers", "retracted: 'the claim fails' framing"),
    (r"Carlsmith median ÷", "retracted: single-figure human comparison"),
]
for pattern, why in RETRACTED:
    for name, text in SURFACES.items():
        check_bool(f"prose: {name} free of {why}",
                   re.search(pattern, text, re.I) is None)

# Statements that must be present, because removing one silently restores a
# one-sided reading of the comparison.
REQUIRED = [
    # about.html now carries the comparison alone. It must not restate it as a
    # single ratio: the convention range is what makes the claim honest.
    ("web/about.html", r"four to six orders of magnitude", "the convention range"),
    ("web/about.html", r"979,763,200", "the state figure"),
]
for name, pattern, why in REQUIRED:
    check_bool(f"prose: {name} still states {why}",
               re.search(pattern, SURFACES[name], re.I) is not None)

# Section numbers must be consecutive from 1. Inserting a section and forgetting
# to renumber the rest shipped two sections called "6" on 2026-07-27.
html_secs = [int(m) for m in re.findall(r"<h2>(\d+)\s*·", SURFACES["web/about.html"])]
check_bool("prose: web/about.html section numbers are 1..N consecutive",
           html_secs == list(range(1, len(html_secs) + 1)))

# Every §N cross-reference must point at a section that exists.
refs = {int(m) for m in re.findall(r"§(\d+)", SURFACES["web/about.html"])}
check_bool("prose: web/about.html §-references all resolve",
           refs.issubset(set(html_secs)))

# The assertion count is quoted in four places. Quoting a number that has
# drifted is exactly the failure this script exists to prevent, so it checks
# its own. This must be the LAST check added: it counts itself.
# ARCHITECTURE.md was left out of this list until 2026-07-28 and drifted to a
# stale count precisely because the guard could not see it.
STATED_IN = ["README.md"]
total_with_this = len(results) + len(STATED_IN)  # this loop adds one per surface
for name in STATED_IN:
    nums = {int(n) for n in re.findall(r"(\d+)\s*(?:\n\s*)?assertions", SURFACES[name])}
    check_bool(f"prose: {name} quotes the real assertion count ({total_with_this})",
               nums == {total_with_this})

# ══════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════

failed = [r for r in results if not r[0]]
width = max(len(r[1]) for r in results)
print(f"\n{'claim'.ljust(width)}  {'measured':>18s}  {'asserted':>22s}")
print("─" * (width + 46))
for ok, label, got, exp in results:
    print(f"{'  ' if ok else '!!'}{label.ljust(width - 2)}  {got:>18s}  {exp:>22s}")

print(f"\n{len(results) - len(failed)} of {len(results)} claims verified.")
if failed:
    print(f"\n{len(failed)} FAILED — update web/js/tours.js or fix the pipeline:")
    for _, label, got, exp in failed:
        print(f"  · {label}: measured {got}, tour says {exp}")
    sys.exit(1)
print("All tour claims hold.")
