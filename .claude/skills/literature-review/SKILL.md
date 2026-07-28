---
name: literature-review
description: "Conduct targeted literature reviews for computational and interpretability research. Search for relevant prior work, identify key papers, and synthesize findings thematically. Lighter-weight than systematic reviews — focused on grounding research claims."
allowed-tools: [Read, Write, Edit, Bash, WebSearch, WebFetch]
---

# Literature Review

## Overview

Conduct targeted literature searches to ground research claims in prior work. This skill is designed for computational research where the goal is to contextualize findings, identify relevant methods, and find competing approaches — not to conduct exhaustive systematic reviews.

## When to Use This Skill

- Grounding a new finding in existing literature
- Identifying relevant prior work before writing an introduction
- Finding methodological precedents for analysis choices
- Checking whether a finding has been reported before
- Building a discussion section that engages with the field

## Workflow

### 1. Define the Search Scope

What specific claims or methods need literature support?
- "Has anyone measured tonic affect baselines in LLMs?"
- "What methods exist for extracting semantic dimensions from activations?"
- "How do MoE and dense architectures differ in representational structure?"

### 2. Search Strategy

**Primary sources:**
- arXiv (cs.CL, cs.LG, cs.AI for NLP/ML interpretability)
- Semantic Scholar API for citation networks
- Google Scholar for broad coverage
- Conference proceedings (NeurIPS, ICML, ICLR, ACL, EMNLP)

**Search tactics:**
- Start broad, then narrow with specific terms
- Follow citation chains (forward and backward)
- Check related work sections of closely relevant papers
- Search for methods, not just topics (e.g., "probing classifiers" not just "sentiment in LLMs")

### 3. Prioritize Papers

**Include if:**
- Directly addresses the same question or method
- Published in a top venue or highly cited (100+ for papers 3+ years old)
- Provides a methodological foundation for your approach
- Reports a conflicting finding (important for honest discussion)

**De-prioritize if:**
- Tangentially related without direct bearing on claims
- Unpublished with no citations and untested methods
- Duplicates a finding already covered by a better paper

### 4. Synthesize Thematically

Organize by research question or theme, not by paper:

**Good**: "Several approaches have been used to probe affect in language models. Smith et al. (2023) used probing classifiers on hidden states, finding... Jones et al. (2024) instead used representation similarity analysis, showing..."

**Bad**: "Smith et al. (2023) studied affect in LLMs. They found... Jones et al. (2024) also studied affect. They found..."

### 5. Document for Reproducibility

For each search:
- Record the query terms used
- Note which databases were searched
- Record the date of search
- Keep track of papers considered and rejected

## Key Research Areas for Our Work

These are the primary literature domains relevant to Phosphenes and related projects:

### Mechanistic Interpretability
- Probing classifiers, linear probes
- Representation geometry (superposition, polytopes)
- Activation patching and causal tracing
- Logit lens and tuned lens
- Sparse autoencoders for feature discovery

### Model Representations and Affect
- Sentiment probing in hidden states
- Emotional content in language model activations
- Self-referential processing in LLMs
- Theory of mind in language models

### Dimensionality and Projection Methods
- Johnson-Lindenstrauss lemma and random projection
- PCA and ICA for neural network analysis
- Effective dimensionality of neural representations
- Intrinsic dimension estimation

### Architecture and Behavior
- MoE vs. dense architecture comparisons
- Scaling laws and emergent capabilities
- Training data effects on model behavior
- Architecture-specific representation patterns

## Citation Format

For technical reports, use author-year inline citations:
- (Smith et al., 2023)
- Smith et al. (2023) showed that...
- Multiple sources: (Smith, 2023; Jones et al., 2024)

Include a full reference list at the end with:
- Authors, Year, Title, Venue/Journal, DOI or URL

## Quality Indicators

A good literature grounding:
- Cites 10-20 papers for a technical report (not 0, not 100)
- Includes both supporting and conflicting findings
- Covers methods, not just results
- References recent work (last 3-5 years) and foundational work
- Identifies what's genuinely new about your contribution
