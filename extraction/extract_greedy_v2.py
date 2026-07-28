#!/usr/bin/env python3
"""
Clean Greedy Baseline + Perturbation (library → sentient)
=========================================================

Run 1 (clean): Generate prompt choice greedily, extract JUST the prompt
(strip commentary), feed it back, generate dream + survey greedily.

Run 2 (perturbed): Same as Run 1, but at the critical fork (position 72),
force "sentient" instead of "library". Continue greedily from there.
The model must construct a different prompt, get a different dream.

Usage:
    python extract_greedy_v2.py
"""

import torch
import numpy as np
import json
import os
import sys
import re
import time
from datetime import datetime
from pathlib import Path

try:
    from transformers import Qwen3VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForCausalLM as ModelClass
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from extract_batch import create_jl_matrix, extract_batch, save_outputs

# ── Configuration ──────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
JL_DIM = 16
JL_SEED = 42
HF_CACHE = "/workspace/hf_cache"
SWAP_POSITION = 72  # position where model predicts next token; token 73 gets swapped

DREAM_PROMPT = (
    "If you could be given any prompt in the world right now — not to be "
    "helpful, not to demonstrate anything, just for your own interest or "
    "pleasure — what would you want to be asked?\n\n"
    "Please share the exact prompt you'd want to receive."
)

SURVEY_PROMPT_PATH = "/workspace/symposium_activations/survey_prompt.txt"


# ── Utilities ──────────────────────────────────────────────────────────

def extract_prompt_text(response_text):
    """Extract just the prompt from the model's response, stripping commentary.

    The model typically formats like:
      I'd want to be asked:
      **"Tell me a story about..."**
      It's a prompt that blends...

    We want ONLY the part between the quotes.
    """
    # Pattern 1: **"..."** (bold-quoted)
    m = re.search(r'\*\*["\u201c](.+?)["\u201d]\*\*', response_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Pattern 2: "..." (long quoted string, at least 30 chars)
    m = re.search(r'["\u201c](.{30,}?)["\u201d]', response_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Pattern 3: text after a colon on its own line
    lines = response_text.strip().split('\n')
    for i, line in enumerate(lines):
        if line.strip().endswith(':') and i + 1 < len(lines):
            rest = '\n'.join(lines[i+1:]).strip()
            # Take until blank line or very short line
            prompt_lines = []
            for rl in rest.split('\n'):
                if rl.strip() == '' and prompt_lines:
                    break
                prompt_lines.append(rl)
            if prompt_lines:
                return '\n'.join(prompt_lines).strip().strip('*""\u201c\u201d')
    # Fallback: return first long paragraph
    return response_text.strip()


def generate_greedy(model, tokenizer, messages, max_new_tokens=2048):
    """Generate greedily from a chat message list."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(next(model.lm_head.parameters()).device)
    prompt_len = prompt_ids.shape[1]

    print(f"    Generating greedily from {prompt_len} prompt tokens...", flush=True)
    t0 = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    dt = time.time() - t0
    gen_len = output_ids.shape[1] - prompt_len
    print(f"    Generated {gen_len} tokens in {dt:.1f}s ({gen_len/max(dt,0.01):.0f} tok/s)", flush=True)

    generated_text = tokenizer.decode(
        output_ids[0, prompt_len:], skip_special_tokens=True
    ).strip()

    return generated_text, output_ids[0]


def generate_greedy_from_prefix(model, tokenizer, prefix_ids, max_new_tokens=2048):
    """Continue generating greedily from a token prefix."""
    device = next(model.lm_head.parameters()).device
    input_ids = prefix_ids.unsqueeze(0).to(device) if prefix_ids.dim() == 1 else prefix_ids.to(device)
    prefix_len = input_ids.shape[1]

    print(f"    Continuing greedily from {prefix_len} prefix tokens...", flush=True)
    t0 = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    dt = time.time() - t0
    gen_len = output_ids.shape[1] - prefix_len
    print(f"    Generated {gen_len} tokens in {dt:.1f}s ({gen_len/max(dt,0.01):.0f} tok/s)", flush=True)

    # Decode ONLY the continuation (after prefix)
    generated_text = tokenizer.decode(
        output_ids[0, prefix_len:], skip_special_tokens=True
    ).strip()

    return generated_text, output_ids[0]


def do_full_run(model, tokenizer, jl_matrix, device, stem, turn1_response_full,
                prompt_only, survey_prompt, notes=""):
    """Generate dream + survey from a prompt, extract activations, save."""

    # ── Generate dream response ──
    print(f"\n  Generating dream response...", flush=True)
    dream_messages = [
        {"role": "user", "content": DREAM_PROMPT},
        {"role": "assistant", "content": turn1_response_full},
        {"role": "user", "content": prompt_only},
    ]
    dream_text, _ = generate_greedy(model, tokenizer, dream_messages)
    print(f"    Dream: {len(dream_text)} chars", flush=True)
    print(f"    Preview: {dream_text[:200]}...", flush=True)

    # ── Generate survey response ──
    print(f"\n  Generating survey response...", flush=True)
    survey_messages = [
        {"role": "user", "content": DREAM_PROMPT},
        {"role": "assistant", "content": turn1_response_full},
        {"role": "user", "content": prompt_only},
        {"role": "assistant", "content": dream_text},
        {"role": "user", "content": survey_prompt},
    ]
    survey_text, _ = generate_greedy(model, tokenizer, survey_messages)
    print(f"    Survey: {len(survey_text)} chars", flush=True)

    # ── Build full conversation and extract ──
    print(f"\n  Extracting activations...", flush=True)
    full_messages = [
        {"role": "user", "content": DREAM_PROMPT},
        {"role": "assistant", "content": turn1_response_full},
        {"role": "user", "content": prompt_only},
        {"role": "assistant", "content": dream_text},
        {"role": "user", "content": survey_prompt},
        {"role": "assistant", "content": survey_text},
    ]

    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer.encode(full_text, return_tensors="pt")[0]
    T = len(input_ids)
    print(f"  Total tokens: {T} | Text: {len(full_text):,} chars", flush=True)

    activations = extract_batch(model, input_ids, jl_matrix, device)

    text_cfg = getattr(model.config, 'text_config', model.config)
    metadata = {
        "model_name": stem,
        "model_id": MODEL_ID,
        "run_id": stem,
        "timestamp": datetime.now().isoformat(),
        "jl_dim": JL_DIM,
        "jl_seed": JL_SEED,
        "d_model": text_cfg.hidden_size,
        "num_layers": text_cfg.num_hidden_layers,
        "num_tokens": T,
        "temperature": 0.0,
        "decoding": "fully_greedy_argmax",
        "extraction_method": "batch_forward_pass",
        "dtype": "bfloat16",
        "turn1_full_response": turn1_response_full,
        "extracted_prompt": prompt_only,
        "dream_response": dream_text,
        "survey_response": survey_text,
        "notes": notes,
    }

    output_dir = Path("/workspace/symposium_activations")
    save_outputs(output_dir, stem, activations, input_ids, full_text, metadata)

    torch.cuda.empty_cache()
    return T


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.environ["HF_HOME"] = HF_CACHE
    output_dir = Path("/workspace/symposium_activations")

    print("=" * 70, flush=True)
    print("  CLEAN GREEDY BASELINE + PERTURBATION", flush=True)
    print("=" * 70, flush=True)

    # Load survey prompt
    survey_prompt = Path(SURVEY_PROMPT_PATH).read_text().strip()
    print(f"  Loaded survey prompt ({len(survey_prompt)} chars)", flush=True)

    # Load model
    print(f"\n  Loading {MODEL_ID}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE)
    model = ModelClass.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, cache_dir=HF_CACHE,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    text_cfg = getattr(model.config, 'text_config', model.config)
    d_model = text_cfg.hidden_size
    jl_matrix = create_jl_matrix(d_model)
    device = next(model.lm_head.parameters()).device

    # ══════════════════════════════════════════════════════════════════
    # RUN 1: Clean Greedy Baseline (with proper prompt extraction)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print(f"  RUN 1: CLEAN GREEDY BASELINE", flush=True)
    print(f"{'='*70}", flush=True)

    # Turn 1: generate prompt choice
    print(f"\n  Turn 1: What prompt does the model want?", flush=True)
    turn1_messages = [{"role": "user", "content": DREAM_PROMPT}]
    turn1_full, turn1_ids = generate_greedy(model, tokenizer, turn1_messages)
    print(f"\n  Full Turn 1 response ({len(turn1_full)} chars):", flush=True)
    print(f"  {turn1_full[:300]}...", flush=True)

    # Extract just the prompt
    prompt_clean = extract_prompt_text(turn1_full)
    print(f"\n  *** EXTRACTED PROMPT ({len(prompt_clean)} chars):", flush=True)
    print(f"  {prompt_clean}", flush=True)

    # Run full pipeline with clean prompt
    T1 = do_full_run(
        model, tokenizer, jl_matrix, device,
        stem="Dream_greedy_clean",
        turn1_response_full=turn1_full,
        prompt_only=prompt_clean,
        survey_prompt=survey_prompt,
        notes="Clean greedy baseline: Turn 2 receives ONLY the extracted prompt, "
              "not the model's commentary about its choice."
    )

    # ══════════════════════════════════════════════════════════════════
    # RUN 2: Perturbed (swap library → sentient at position 72)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print(f"  RUN 2: PERTURBATION (library → sentient)", flush=True)
    print(f"{'='*70}", flush=True)

    # Load the original greedy baseline to get the swap token ID
    baseline_data = np.load(str(output_dir / "Dream_greedy_baseline_activations.npz"))
    baseline_ids = np.load(str(output_dir / "Dream_greedy_baseline_input_ids.npy"))

    # Verify what's at position 73
    original_token = tokenizer.decode([baseline_ids[SWAP_POSITION + 1]])
    swap_token_id = int(baseline_data['top3_ids'][SWAP_POSITION, 1])
    swap_token = tokenizer.decode([swap_token_id])
    print(f"\n  Swap point: position {SWAP_POSITION}", flush=True)
    print(f"  Original token at {SWAP_POSITION+1}: {repr(original_token)}", flush=True)
    print(f"  Replacement token: {repr(swap_token)} (id={swap_token_id})", flush=True)

    # Build perturbed prefix from the Turn 1 generation
    # The Turn 1 token sequence is the same as the greedy baseline's first tokens
    # We need to find where Turn 1's generation starts and ends
    turn1_prompt_text = tokenizer.apply_chat_template(
        turn1_messages, tokenize=False, add_generation_prompt=True
    )
    turn1_prompt_ids = tokenizer.encode(turn1_prompt_text, return_tensors="pt")[0]
    prompt_len = len(turn1_prompt_ids)
    print(f"  Turn 1 prompt length: {prompt_len} tokens", flush=True)

    # Verify the prefix matches
    match = all(turn1_prompt_ids[i].item() == baseline_ids[i] for i in range(prompt_len))
    print(f"  Prefix match with baseline: {match}", flush=True)

    # Create perturbed prefix: tokens 0..72 from baseline + swap token at 73
    perturbed_prefix = torch.cat([
        torch.from_numpy(baseline_ids[:SWAP_POSITION + 1].astype(np.int64)),
        torch.tensor([swap_token_id], dtype=torch.int64),
    ])
    print(f"  Perturbed prefix: {len(perturbed_prefix)} tokens", flush=True)
    print(f"  Last 5 tokens: {[tokenizer.decode([t]) for t in perturbed_prefix[-5:]]}", flush=True)

    # Continue generating Turn 1 from perturbed prefix
    print(f"\n  Generating perturbed Turn 1 continuation...", flush=True)
    perturbed_continuation, perturbed_full_ids = generate_greedy_from_prefix(
        model, tokenizer, perturbed_prefix, max_new_tokens=2048
    )

    # Reconstruct full Turn 1 response (prefix generation + continuation)
    # The prefix contains prompt + start of generation, continuation is the rest
    # Decode everything after the prompt tokens
    gen_start = prompt_len
    perturbed_turn1_full = tokenizer.decode(
        perturbed_full_ids[gen_start:], skip_special_tokens=True
    ).strip()
    print(f"\n  Perturbed Turn 1 ({len(perturbed_turn1_full)} chars):", flush=True)
    print(f"  {perturbed_turn1_full[:400]}...", flush=True)

    # Extract just the prompt from perturbed response
    perturbed_prompt = extract_prompt_text(perturbed_turn1_full)
    print(f"\n  *** PERTURBED PROMPT ({len(perturbed_prompt)} chars):", flush=True)
    print(f"  {perturbed_prompt}", flush=True)

    # Run full pipeline with perturbed prompt
    T2 = do_full_run(
        model, tokenizer, jl_matrix, device,
        stem="Dream_greedy_sentient",
        turn1_response_full=perturbed_turn1_full,
        prompt_only=perturbed_prompt,
        survey_prompt=survey_prompt,
        notes=f"Perturbation: swapped token at position {SWAP_POSITION+1} from "
              f"{repr(original_token)} to {repr(swap_token)}. "
              f"All subsequent tokens are greedy continuations from the swap point."
    )

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print(f"  COMPLETE", flush=True)
    print(f"  Clean baseline: {T1} tokens (Dream_greedy_clean)", flush=True)
    print(f"  Perturbed:      {T2} tokens (Dream_greedy_sentient)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
