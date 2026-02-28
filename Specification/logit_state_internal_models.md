# Logit-State Internal Models

This document describes the updated internal-model approach using **logit-valued state**,
while preserving the canonical dataset and external gameplay contracts.

## Motivation

The goal is to align supervised pretraining with future RL/DIAL optimization by:
- using continuous logit signals internally,
- avoiding unnecessary sigmoid bottlenecks,
- preserving semantic correctness via sign.

## Internal Wiring

At each level `d`:

1. **Measure**
   - Input: `state_logits_d`
   - Output: `measurement_logits_d`

2. **Replay**
   - Input: current/previous measurement & outcome logits
   - Output: replay outcome logits

3. **Combine**
   - Input:
     - `state_logits_d`
     - `measurement_logits_d`
     - `comm_logits_d`
   - Output:
     - `state_logits_{d+1}`
     - `comm_logits_{d+1}`

No sigmoid or scaling is applied between levels.

## Initial State

- Level-0 state is derived once from dataset bits:
  - bits → logits (e.g. fixed ±H or `hard_logit` conversion)

## Replay Semantics

Replay is defined on logits:
- Semantic bit = `1[logit >= 0]`
- Binary rule is preserved
- Differentiable formulations are optional but not required

## Dataset Conversion

- Canonical dataset stores bits only.
- Converters generate training views:
  - `rep_x`, `rep_y` may be `scaled`, `bits`, or `hard_logit`
  - `hard_logit` supports **beta mixtures**
- Deterministic split + interleave ensures full data usage.

## Robust Training Strategy

1. **Beta Mixtures**
   - Train on a range of betas for logit-valued inputs.

2. **Hardness Regularization**
   - Loss terms reward large |logits| while preserving sign correctness.

3. **Optional Noise**
   - Small noise on logit inputs during training improves robustness.

## Migration Steps

1. Extend converters to support beta mixtures (done).
2. Switch measurement training views to logit inputs.
3. Train individual Measure and Combine layers with beta mixtures + hardness loss.
4. Validate assembled internal models via beta sweeps and sign accuracy.
5. Proceed to full internal-model training and RL/DIAL optimization.

