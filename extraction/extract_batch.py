#!/usr/bin/env python3
"""
Batch Activation Extraction — CIMC Latent Journey
==================================================

Fast extraction using single forward pass per conversation.
Because of causal attention masks, hidden states at each position
are identical to what they were during autoregressive generation.

Extracts:
  - JL projections & movement metrics (per layer)
  - Token entropy & hidden state norms (per layer)
  - Logit lens: per-layer entropy & rank of actual token
  - Prediction analysis: top-3 tokens, actual token rank & probability

Usage:
    python extract_batch.py
    python extract_batch.py --conversations /path/to/extraction_conversations.json
    python extract_batch.py --conv-id conv_00173  # Extract one only
"""

import torch
import numpy as np
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer
try:
    from transformers import Qwen3VLForConditionalGeneration as ModelClass
except ImportError:
    from transformers import AutoModelForCausalLM as ModelClass

# ── Configuration ──────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
JL_DIM = 16
JL_SEED = 42
HF_CACHE = "/workspace/hf_cache"

# ── JL Projection ─────────────────────────────────────────────────────

def create_jl_matrix(d_model, k=JL_DIM, seed=JL_SEED):
    """Create Rademacher JL projection matrix (+-1 scaled by 1/sqrt(k))."""
    rng = np.random.RandomState(seed)
    R = rng.choice([-1, 1], size=(d_model, k)).astype(np.float16)
    R = R / np.sqrt(k)
    return torch.from_numpy(R)


# ── Batch Extraction ──────────────────────────────────────────────────

def extract_batch(model, input_ids, jl_matrix, device):
    """
    Extract activations via single forward pass.

    Because of causal attention masks, hidden states at position t
    are identical whether computed autoregressively or in a batch.
    This is orders of magnitude faster than token-by-token.
    """
    T = len(input_ids)
    text_cfg = getattr(model.config, 'text_config', model.config)
    L = text_cfg.num_hidden_layers
    K = jl_matrix.shape[1]
    d_model = text_cfg.hidden_size

    R = jl_matrix.to(device=device, dtype=torch.float32)

    print(f"    Forward pass: {T} tokens x {L} layers...")
    t0 = time.time()

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0).to(device),
            output_hidden_states=True,
            return_dict=True,
        )

    dt_forward = time.time() - t0
    print(f"    Forward pass complete: {dt_forward:.1f}s ({T/dt_forward:.0f} tok/s)")

    # Extract logits for token entropy
    logits = outputs.logits[0]  # (T, vocab_size)
    hidden_states = outputs.hidden_states  # tuple of (1, T, d_model), length L+1

    # ── Final-layer token entropy ──
    # Cast to float32 for numerical stability and numpy compatibility
    print(f"    Computing final-layer token entropy...")
    logits_f = logits.float()
    log_probs = torch.log_softmax(logits_f, dim=-1)
    probs = torch.softmax(logits_f, dim=-1)
    token_entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy().astype(np.float32)  # (T,)

    # ── Prediction analysis (final layer) ──
    # logits[t] predicts token at position t+1, so we use logits[:-1] vs input_ids[1:]
    print(f"    Computing prediction analysis (top-3, actual rank)...")
    next_token_ids = input_ids[1:].to(device)  # (T-1,) actual next tokens
    pred_logits = logits_f[:-1]  # (T-1, vocab_size) already float32

    # Top-3 predicted tokens
    top3_vals, top3_ids_t = torch.topk(pred_logits, k=3, dim=-1)  # (T-1, 3)
    pred_probs = torch.softmax(pred_logits, dim=-1)
    top3_probs_t = pred_probs.gather(1, top3_ids_t)  # (T-1, 3)

    # Rank and probability of actual next token
    actual_logit = pred_logits.gather(1, next_token_ids.unsqueeze(1))  # (T-1, 1)
    actual_rank = (pred_logits > actual_logit).sum(dim=-1)  # (T-1,) 0-indexed rank
    actual_prob = pred_probs.gather(1, next_token_ids.unsqueeze(1)).squeeze(-1)  # (T-1,)

    # Move to CPU/numpy
    top3_ids_np = top3_ids_t.cpu().numpy().astype(np.int32)      # (T-1, 3)
    top3_probs_np = top3_probs_t.cpu().numpy().astype(np.float32)  # (T-1, 3)
    actual_rank_np = actual_rank.cpu().numpy().astype(np.int32)    # (T-1,)
    actual_prob_np = actual_prob.cpu().numpy().astype(np.float32)  # (T-1,)

    del logits_f, pred_logits, pred_probs  # free memory

    # ── Compute per-layer metrics ──
    print(f"    Computing JL projections and movement metrics...")
    t0_metrics = time.time()

    jl_sketches = np.zeros((T, L, K), dtype=np.float16)
    delta_l2 = np.zeros((T, L), dtype=np.float32)
    top1_frac = np.zeros((T, L), dtype=np.float32)
    top25_frac = np.zeros((T, L), dtype=np.float32)
    cos_prev = np.zeros((T, L), dtype=np.float32)
    jl_energy = np.zeros((T, L), dtype=np.float32)
    h_norm = np.zeros((T, L), dtype=np.float32)
    logit_lens_entropy = np.zeros((T, L), dtype=np.float32)
    logit_lens_rank = np.zeros((T - 1, L), dtype=np.int32)

    # Get final layernorm and lm_head for logit lens
    # Qwen3-VL hierarchy: model.model.language_model.norm + model.lm_head
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    else:
        lm_head = model.model.lm_head
    if hasattr(model.model, 'language_model'):
        final_norm = model.model.language_model.norm  # Qwen3-VL
    elif hasattr(model.model, 'norm'):
        final_norm = model.model.norm  # Standard causal LM
    else:
        raise RuntimeError("Could not find final norm layer")

    for l in range(L):
        # Hidden states for all tokens at this layer: (T, d_model)
        H = hidden_states[l + 1][0].float()  # (T, d_model)

        # Hidden state norms
        h_norm[:, l] = torch.norm(H, dim=-1).cpu().numpy()

        # JL projection: (T, d_model) @ (d_model, K) -> (T, K)
        Y = H @ R  # (T, K)
        jl_sketches[:, l, :] = Y.cpu().numpy().astype(np.float16)
        jl_energy[:, l] = torch.norm(Y, dim=-1).cpu().numpy()

        # Movement metrics (token-to-token differences)
        if T > 1:
            delta = H[1:] - H[:-1]  # (T-1, d_model)
            s = delta ** 2  # (T-1, d_model)
            total = s.sum(dim=-1)  # (T-1,)

            # delta_l2
            delta_l2[1:, l] = torch.sqrt(total).cpu().numpy()

            # top1_frac and top25_frac
            k1 = max(1, int(np.ceil(0.01 * d_model)))
            k25 = max(1, int(np.ceil(0.25 * d_model)))

            sorted_s, _ = torch.sort(s, dim=-1, descending=True)  # (T-1, d_model)

            mask = total > 1e-10
            if mask.any():
                top1 = sorted_s[:, :k1].sum(dim=-1)
                top25 = sorted_s[:, :k25].sum(dim=-1)

                safe_total = total.clone()
                safe_total[~mask] = 1.0  # avoid division by zero

                top1_vals = (top1 / safe_total).cpu().numpy()
                top25_vals = (top25 / safe_total).cpu().numpy()
                top1_vals[~mask.cpu().numpy()] = 0.0
                top25_vals[~mask.cpu().numpy()] = 0.0

                top1_frac[1:, l] = top1_vals
                top25_frac[1:, l] = top25_vals

            # cosine similarity with previous token
            cos = torch.nn.functional.cosine_similarity(H[1:], H[:-1], dim=-1)
            cos_prev[1:, l] = cos.cpu().numpy()

        # ── Logit lens at this layer ──
        with torch.no_grad():
            # Apply final layernorm + lm_head to this layer's hidden states
            H_normed = final_norm(H.to(next(final_norm.parameters()).dtype))
            ll_logits = lm_head(H_normed).float()  # (T, vocab_size) cast to f32

            # Entropy of logit lens predictions
            ll_probs = torch.softmax(ll_logits, dim=-1)
            ll_log_probs = torch.log_softmax(ll_logits, dim=-1)
            logit_lens_entropy[:, l] = -(ll_probs * ll_log_probs).sum(dim=-1).cpu().numpy()

            # Rank of actual next token at this layer
            if T > 1:
                ll_actual_logit = ll_logits[:-1].gather(1, next_token_ids.unsqueeze(1))  # (T-1, 1)
                logit_lens_rank[:, l] = (ll_logits[:-1] > ll_actual_logit).sum(dim=-1).cpu().numpy()

        del H_normed, ll_logits, ll_probs, ll_log_probs
        torch.cuda.empty_cache()

        if (l + 1) % 8 == 0:
            print(f"      Layer {l+1}/{L} done")

    dt_metrics = time.time() - t0_metrics
    print(f"    Metrics complete: {dt_metrics:.1f}s")

    return {
        # Per-layer metrics: (T, L) or (T, L, K)
        "jl": jl_sketches,             # (T, L, 16) JL sketches
        "delta_l2": delta_l2,           # (T, L) movement magnitude
        "top1_frac": top1_frac,         # (T, L) concentration in top 1% dims
        "top25_frac": top25_frac,       # (T, L) concentration in top 25% dims
        "cos_prev": cos_prev,           # (T, L) cosine sim with previous token
        "jl_energy": jl_energy,         # (T, L) JL projection norm
        "h_norm": h_norm,               # (T, L) hidden state norm
        "logit_lens_entropy": logit_lens_entropy,  # (T, L) entropy at each layer
        "logit_lens_rank": logit_lens_rank,        # (T-1, L) rank of actual next token
        # Final-layer prediction analysis
        "token_entropy": token_entropy,   # (T,) final-layer entropy
        "top3_ids": top3_ids_np,          # (T-1, 3) top 3 predicted token IDs
        "top3_probs": top3_probs_np,      # (T-1, 3) their probabilities
        "actual_rank": actual_rank_np,    # (T-1,) rank of actual next token
        "actual_prob": actual_prob_np,    # (T-1,) probability of actual next token
    }


# ── Save Outputs ──────────────────────────────────────────────────────

def save_outputs(output_dir, stem, activations, input_ids, full_text, metadata):
    output_dir = Path(output_dir)

    npz_path = output_dir / f"{stem}_activations.npz"
    np.savez_compressed(str(npz_path), **activations)
    npz_size = npz_path.stat().st_size / 1024 / 1024
    print(f"    Saved: {npz_path.name} ({npz_size:.1f}MB)")

    npy_path = output_dir / f"{stem}_input_ids.npy"
    np.save(str(npy_path), input_ids.numpy())

    txt_path = output_dir / f"{stem}_text.txt"
    txt_path.write_text(full_text)

    meta_path = output_dir / f"{stem}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"    Saved: {meta_path.name}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Activation Extraction")
    parser.add_argument("--conversations", type=str,
                        default="/workspace/symposium_activations/extraction_conversations.json")
    parser.add_argument("--output", type=str,
                        default="/workspace/symposium_activations")
    parser.add_argument("--hf-cache", type=str, default=HF_CACHE)
    parser.add_argument("--conv-id", type=str, default=None,
                        help="Extract only this conversation ID")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  BATCH ACTIVATION EXTRACTION — CIMC Latent Journey")
    print("=" * 70)

    # Load conversations
    conversations = json.loads(Path(args.conversations).read_text())
    if args.conv_id:
        conversations = [c for c in conversations if c['conversation_id'] == args.conv_id]
    print(f"\n  Conversations to extract: {len(conversations)}")

    # Load model
    print(f"\n  Loading {MODEL_ID} in bfloat16...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=args.hf_cache
    )
    model = ModelClass.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.hf_cache,
    )

    dt = time.time() - t0
    # Qwen3-VL nests text config under .text_config
    text_cfg = getattr(model.config, 'text_config', model.config)
    d_model = text_cfg.hidden_size
    num_layers = text_cfg.num_hidden_layers
    print(f"  Loaded in {dt:.1f}s | d_model={d_model}, layers={num_layers}")

    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {used:.1f}GB / {total_mem:.1f}GB")

    # Create JL matrix
    jl_matrix = create_jl_matrix(d_model)
    print(f"  JL matrix: ({d_model}, {JL_DIM}) | seed={JL_SEED}")

    # Extract each conversation
    results = {}
    for conv in conversations:
        conv_id = conv['conversation_id']
        print(f"\n{'='*70}")
        print(f"  Extracting: {conv_id}")
        print(f"{'='*70}")

        # Tokenize using chat template (reproduces exact generation context)
        full_text = tokenizer.apply_chat_template(
            conv['messages'], tokenize=False, add_generation_prompt=False
        )
        input_ids = tokenizer.encode(full_text, return_tensors="pt")[0]
        T = len(input_ids)
        print(f"  Tokens: {T} | Text: {len(full_text):,} chars")

        # Check if sequence fits in memory
        # Rough estimate: hidden states = T * L * d_model * 4 bytes (float32)
        mem_estimate_gb = T * num_layers * d_model * 4 / 1e9
        print(f"  Estimated hidden state memory: {mem_estimate_gb:.1f}GB")

        # Find what device the lm_head is on (reliable even with device_map)
        device = next(model.lm_head.parameters()).device

        # Extract
        activations = extract_batch(model, input_ids, jl_matrix, device)

        # Build stem
        stem = f"Dream_{conv_id}_run1"

        # Metadata
        metadata = {
            "model_name": f"Dream_{conv_id}",
            "model_id": MODEL_ID,
            "conversation_id": conv_id,
            "run_id": 1,
            "timestamp": datetime.now().isoformat(),
            "jl_dim": JL_DIM,
            "jl_seed": JL_SEED,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_tokens": T,
            "original_temperature": conv.get('temperature'),
            "survey_ratings": conv.get('survey_ratings', {}),
            "extraction_method": "batch_forward_pass",
            "dtype": "bfloat16",
            "notes": "Exact reproduction of generation context via chat template. "
                     "Causal mask guarantees hidden states match original generation.",
        }

        # Save
        print(f"\n  Saving outputs...")
        save_outputs(output_dir, stem, activations, input_ids, full_text, metadata)

        results[conv_id] = {"stem": stem, "tokens": T, "layers": num_layers}

        # Clear GPU cache between conversations
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"  EXTRACTION COMPLETE — {len(results)} conversations")
    print(f"{'='*70}")
    for conv_id, info in results.items():
        print(f"  {conv_id}: {info['tokens']} tokens, {info['layers']} layers")


if __name__ == "__main__":
    main()
