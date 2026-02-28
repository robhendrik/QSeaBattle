# API & Behavior Contract (Updated)

This contract preserves the frozen gameplay interface and clarifies internal representations.

## Player Interfaces (Unchanged)

### Player A
- Input: `field_batch` — float32 tensor `{0.0,1.0}`
- Output:
  - `comm_logits`
  - `meas_list`, `out_list` (logits)

### Player B
- Input:
  - `gun_batch`, `comm_batch`, `prev_meas_batch`, `prev_out_batch`
- Output:
  - `shoot_logit`

Adapters hide all internal representation changes.

## Internal Models

- Measure layers: logit outputs
- Combine layers: logit outputs
- Replay: logit-in / logit-out

State is logit-valued between levels.

## Adapters

- Input bits are mapped to logits at entry.
- Outputs are mapped to bits/probabilities only at gameplay boundary.
- No external contract changes.

