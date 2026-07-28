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
check("step1: growth factor layer 0 -> 60", float(energy[60] / energy[0]), 78.0, 2.0, "x")

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — seams fire at turn ENDINGS, not beginnings
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

check("step2: mean seam at <|im_end|>", end_m, 0.780, 0.02)
check("step2: mean seam elsewhere", other_m, 0.110, 0.02)
check("step2: mean seam at <|im_start|>", start_m, 0.004, 0.02)
check("step2: im_end / elsewhere ratio", float(ratios.mean()), 7.1, 0.8, "x")
check("step2: ratio range low", float(ratios.min()), 6.2, 0.4, "x")
check("step2: ratio range high", float(ratios.max()), 7.8, 0.9, "x")
check_bool("step2: im_end > im_start in all 8 sessions",
           bool(np.all(np.array(end_means) > np.array(start_means))))

flag = bundle(FLAGSHIP)
seam_flag = u8(flag, "seam_score").astype(np.float32) / 255.0
check_bool("step2: token 0 seam suppressed to zero", float(seam_flag[0]) == 0.0)
check("step2: seam at token 193 (im_end)", float(seam_flag[193]), 0.972, 0.02)

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

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — logit-lens entropy rises then falls
# ══════════════════════════════════════════════════════════════════════

ent = a["logit_lens_entropy"].astype(np.float32).mean(axis=0)
peak = int(np.argmax(ent[:24]))
check("step4: entropy at layer 0", float(ent[0]), 8.84, 0.1, " nats")
check("step4: entropy peak layer", peak, 9, 1)
check("step4: entropy at peak", float(ent[peak]), 9.84, 0.05, " nats")
check("step4: entropy at final layer", float(ent[-1]), 1.65, 0.1, " nats")
check("step4: ln(vocab) reference", float(np.log(151936)), 11.93, 0.01, " nats")
check_bool("step4: entropy rises before it falls", bool(ent[peak] > ent[0] and ent[-1] < ent[0]))

rise_in_all = sum(
    1 for stem in ALL_STEMS
    if (lambda e: e[int(np.argmax(e[:24]))] > e[0] and e[-1] < e[0])(
        npz(stem)["logit_lens_entropy"].astype(np.float32).mean(axis=0))
)
check("step4: sessions showing the rise-then-fall", rise_in_all, 8, 0)

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
check("fork: typical JL magnitude", float(np.linalg.norm(jl_l, axis=2).mean()), 299.2, 4.0)
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
# THE COMPUTE ARGUMENT — every figure quoted in docs/THESIS.md §1-§3 and
# web/about.html §1-§3, recomputed from the recorded architecture.
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
# PROSE CONSISTENCY — docs/THESIS.md and web/about.html are the same
# argument at two lengths. A correction applied to one and not the other
# has already happened once and shipped; this is the regression test.
# ══════════════════════════════════════════════════════════════════════

import re

SURFACES = {
    "docs/THESIS.md": (ROOT / "docs" / "THESIS.md").read_text(),
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
    ("docs/THESIS.md", r"overlap", "the overlap verdict"),
    ("docs/THESIS.md", r"marginal", "the marginal convention, named"),
    ("docs/THESIS.md", r"full[- ]context", "the full-context convention, named"),
    ("docs/THESIS.md", r"efficiency, not shortfall", "the deflationary-framing caveat"),
    ("web/about.html", r"overlap", "the overlap verdict"),
    ("web/about.html", r"marginal", "the marginal convention, named"),
    ("web/about.html", r"full[- ]context", "the full-context convention, named"),
    ("web/about.html", r"efficiency, not shortfall", "the deflationary-framing caveat"),
    # The corpus was elicited under a leading prompt. Disclosed, or it is not honest.
    ("web/about.html", r"Recent research has confirmed", "the demand-characteristic disclosure"),
    ("README.md", r"Recent research has confirmed", "the demand-characteristic disclosure"),
    ("README.md", r"refusal", "the refusal session, acknowledged"),
]
for name, pattern, why in REQUIRED:
    check_bool(f"prose: {name} still states {why}",
               re.search(pattern, SURFACES[name], re.I) is not None)

# Section numbers must be consecutive from 1. Inserting a section and forgetting
# to renumber the rest shipped two sections called "6" on 2026-07-27.
html_secs = [int(m) for m in re.findall(r"<h2>(\d+)\s*·", SURFACES["web/about.html"])]
md_secs = [int(m) for m in re.findall(r"^##\s+(\d+)\.", SURFACES["docs/THESIS.md"], re.M)]
check_bool("prose: web/about.html section numbers are 1..N consecutive",
           html_secs == list(range(1, len(html_secs) + 1)))
check_bool("prose: docs/THESIS.md section numbers are 1..N consecutive",
           md_secs == list(range(1, len(md_secs) + 1)))

# Every §N cross-reference must point at a section that exists.
for name, secs in [("web/about.html", html_secs), ("docs/THESIS.md", md_secs)]:
    refs = {int(m) for m in re.findall(r"§(\d+)", SURFACES[name])}
    check_bool(f"prose: {name} §-references all resolve",
               refs.issubset(set(secs)))

# The assertion count is quoted in four places. Quoting a number that has
# drifted is exactly the failure this script exists to prevent, so it checks
# its own. This must be the LAST check added: it counts itself.
STATED_IN = ["README.md", "docs/THESIS.md", "web/about.html"]
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
