---
name: critical-analysis
description: "Self-review research for rigor. Assess methodology, statistical validity, overinterpretation, confounds, and evidence quality. Apply before sharing any research document."
allowed-tools: [Read, Write, Edit, Bash]
---

# Critical Analysis

## Overview

Systematically evaluate research work — your own or others' — for rigor, validity, and honesty. This skill is designed as a self-review checklist to catch common problems before sharing research. It draws on peer review standards but is adapted for computational and interpretability research.

## When to Use This Skill

- Before sharing any research document, report, or analysis
- When reviewing your own methodology and conclusions
- When evaluating claims in papers you're reading
- When a finding seems too clean or too surprising
- When you want to strengthen your own work proactively

## The Self-Review Checklist

### 1. Claims vs. Evidence

For every claim in the document:
- [ ] Is the evidence cited actually in the results section?
- [ ] Does the evidence support the specific claim, or a weaker version?
- [ ] Is the claim appropriately hedged for the strength of evidence?
- [ ] Would a skeptical reader accept this connection?

**Red flags:**
- "This demonstrates that..." (implies proof from correlational data)
- "Clearly..." / "Obviously..." (assertion masquerading as evidence)
- Claims in the discussion that go beyond what the results showed
- Effect described as "large" or "strong" without quantification

### 2. Statistical and Quantitative Rigor

- [ ] Are effect sizes reported alongside significance tests?
- [ ] Are confidence intervals or uncertainty estimates provided?
- [ ] Is the sample size adequate for the claims being made?
- [ ] Were multiple comparisons corrected for?
- [ ] Are the right tests used for the data type?
- [ ] Are assumptions of statistical tests met?
- [ ] Is exploratory analysis clearly labeled as such?

**Red flags:**
- p = 0.049 (suspiciously close to threshold)
- Many tests run, only significant ones reported
- Large effects from very small samples
- Parametric tests on non-normal data without justification

### 3. Confounds and Alternative Explanations

- [ ] What confounds could explain the observed effect?
- [ ] Were appropriate baselines or controls used?
- [ ] Could the result be an artifact of preprocessing?
- [ ] Is the comparison fair (same data, same conditions)?
- [ ] Were important variables controlled or accounted for?

**Common confounds in interpretability research:**
- Token frequency effects (common words behave differently)
- Position effects (early vs. late in sequence)
- Length effects (longer sequences have different properties)
- Model-specific artifacts (architecture, training data)
- Quantization artifacts (uint8 discretization)
- Depth gradient dominance (most variance is layer position)

### 4. Methodology Assessment

- [ ] Could someone reproduce this from the methods section?
- [ ] Are all preprocessing steps documented?
- [ ] Are hyperparameters and thresholds justified?
- [ ] Were results robust to reasonable parameter changes?
- [ ] Is the code available or described in sufficient detail?

### 5. Figures and Presentation

- [ ] Does every figure have a complete, self-contained caption?
- [ ] Are axes labeled with units?
- [ ] Are error bars or uncertainty shown where appropriate?
- [ ] Could the figures be misleading (truncated axes, cherry-picked examples)?
- [ ] Is every figure discussed in the text?
- [ ] Are color choices accessible (colorblind-safe)?

### 6. Intellectual Honesty

- [ ] Are negative results reported?
- [ ] Are limitations discussed prominently, not buried?
- [ ] Is prior work fairly represented?
- [ ] Are the contributions accurately scoped (not overclaimed)?
- [ ] Is speculation clearly distinguished from findings?

### 7. The Adversarial Test

Imagine a skeptical reader. What would they challenge?
- What's the weakest link in the argument?
- What's the most likely boring explanation?
- What control or comparison is missing?
- Where is the reasoning most hand-wavy?

Write down these challenges and either address them in the text or acknowledge them as limitations.

## Severity Classification

When reviewing, classify issues as:

**Critical**: Threatens the validity of main conclusions
- Wrong statistical test for the data type
- Major confound not addressed
- Claims unsupported by evidence
- Reproducibility impossible from description

**Important**: Affects interpretation but not fatally
- Missing baselines or comparisons
- Incomplete uncertainty quantification
- Over-interpretation of marginal effects
- Key methodological details omitted

**Minor**: Worth noting, doesn't change conclusions
- Unclear figure labels
- Imprecise language
- Missing sample sizes in captions
- Stylistic issues

## Output Format

```markdown
## Self-Review Summary

### Overall Assessment
[1-2 sentences on the document's strengths and readiness]

### Critical Issues
[List any critical problems, or "None identified"]

### Important Issues
1. [Issue]: [What to do about it]
2. ...

### Minor Issues
1. [Issue]: [Suggestion]
2. ...

### Adversarial Challenges
1. [What a skeptic would say]: [Your response or acknowledgment]
2. ...

### Verdict
[Ready to share / Needs revision / Major issues to address]
```
