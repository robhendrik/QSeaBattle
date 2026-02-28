# Canonical Specification (Updated)

This specification defines the canonical representations, internal wiring, and training views
for QSeaBattle with **logit-state internal models**.

## Representations

- **Bit**: `{0,1}` semantic value, stored in canonical datasets.
- **Scaled**: numeric representation in `[-0.5,+0.5]`, legacy internal state.
- **Logit**: unbounded real value; semantic bit is `1[logit >= 0]`.

### Canonical rule
All *learned layers* (Measure, Combine, Replay) **produce logits**.
Semantic decisions are based on sign only.

## Internal State Convention (New)

- State (field/gun) **between levels** is represented as **logits**.
- No sigmoid or scaling is applied between Combine → next Measure.
- Level-0 state is adapted once from dataset bits to logits.

This aligns supervised pretraining with RL/DIAL-style continuous internal signaling.

## Replay

Replay is fully **logit-in / logit-out**.
Binary semantics are defined via sign; internal implementation is unchanged.

## Dataset

The canonical dataset remains unchanged (bit storage).
Training views are generated via converters with configurable representations.

## Robustness

- Converters support **beta mixtures** for `hard_logit` representations.
- Training losses may include **hardness regularization** encouraging large |logits|.

