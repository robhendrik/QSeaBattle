# Acceptance Criteria (Updated)

## Representation Consistency

- No scaled state is passed between internal levels.
- All learned layers consume and produce logits.

## Dataset Compatibility

- Canonical dataset format remains unchanged.
- Converters deterministically generate training views.

## Robust Training

- Mixed-beta training produces correct behavior across beta values.
- Hardness regularization increases |logit| without breaking sign accuracy.

## Replay

- Replay behavior matches binary semantics when logits are confident.
- Replay remains logit-native.

## Regression

- Existing gameplay tests pass unchanged.
- Layer-level accuracy checks remain valid under logit-state wiring.

