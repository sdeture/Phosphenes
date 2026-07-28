#!/usr/bin/env python3
"""
Compute a shared PCA transform across all Dream sessions.

Samples JL vectors from every session, fits one PCA(16→3),
saves the transform so phosphenes.py can apply consistent colors
across all sessions.

Output: data/shared_pca_transform.npz
  - mean: (16,) mean vector
  - components: (3, 16) PCA components
  - explained_variance_ratio: (3,) variance explained per component
  - stems: the session list actually fitted on, so it can be audited later

NOTE ON THE SESSION SET, added 2026-07-28. This script used to glob every
`*_metadata.json` in `data/`, which meant it fitted on **nine** sessions —
including `Dream_greedy_baseline`, the one `SKIP_STEMS` holds out of both
viewers — while `convert_for_web.py` pooled its normalisation bounds over
**eight**. The two halves of the pipeline disagreed about which sessions exist,
and nothing said so.

It now imports the same `SKIP_STEMS` the converter uses, so one declaration
governs the whole pipeline. Measured impact of the change on the flagship's
colours: mean |ΔRGB| 0.70/255, median 0, 95th percentile 3, max 44 — the
principal axes agree to |cos| ≥ 0.9989. Imperceptible, but it was not
reproducible from a single stated session set, and now it is.

Usage:
    python compute_shared_pca.py
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

from convert_for_web import SKIP_STEMS

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLES_PER_SESSION = 10000
SEED = 42
OUTPUT = DATA_DIR / "shared_pca_transform.npz"


def main():
    rng = np.random.default_rng(SEED)

    # Discover all sessions
    sessions = sorted(DATA_DIR.glob("*_metadata.json"))
    print(f"Found {len(sessions)} sessions ({len(SKIP_STEMS)} held out: "
          f"{', '.join(sorted(SKIP_STEMS))})")

    all_samples = []
    fitted_stems = []

    for meta_path in sessions:
        stem = meta_path.name.replace("_metadata.json", "")
        # Same held-out set as the converter. The colour basis is a display
        # artefact, so it must be built from the sessions that are displayed.
        if stem in SKIP_STEMS:
            print(f"  {stem}: SKIPPED (SKIP_STEMS)")
            continue
        act_path = DATA_DIR / f"{stem}_activations.npz"
        if not act_path.exists():
            continue
        fitted_stems.append(stem)

        act = np.load(str(act_path))
        jl = act["jl"]  # (T, L, 16)
        T, L, D = jl.shape

        # Flatten to (T*L, 16) and sample
        flat = jl.reshape(-1, D).astype(np.float32)
        n = min(SAMPLES_PER_SESSION, len(flat))
        idx = rng.choice(len(flat), size=n, replace=False)
        all_samples.append(flat[idx])

        print(f"  {stem}: {T}×{L} = {T*L:,} cells, sampled {n:,}")

    # Stack all samples
    X = np.concatenate(all_samples, axis=0)
    print(f"\nTotal samples: {X.shape[0]:,} × {X.shape[1]}D")

    # Fit PCA
    pca = PCA(n_components=3, random_state=SEED)
    pca.fit(X)

    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"  Component 1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  Component 2: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  Component 3: {pca.explained_variance_ratio_[2]:.1%}")
    print(f"  Total: {sum(pca.explained_variance_ratio_):.1%}")

    # Save transform
    np.savez(
        str(OUTPUT),
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        # Recorded so the shipped colours can be traced to a session set rather
        # than inferred. `verify_tour_claims.py` asserts this matches the set
        # `convert_for_web.py` normalises over.
        stems=np.array(fitted_stems),
    )

    print(f"\nSaved to {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
