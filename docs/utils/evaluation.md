# Utility module: Evaluation

## Purpose
Provide helpers for evaluating trained players using tournaments and aggregating results.

## Core utilities

### `evaluate_players(players, layout)`
Run a tournament and return performance statistics.

### `confidence_interval(mean, std_error)`
Compute confidence intervals for reported results.

## Contract
- Evaluation MUST NOT alter player state
- Evaluation MUST be reproducible given fixed seeds

## Invariants
- Results MUST correspond to exactly the games played