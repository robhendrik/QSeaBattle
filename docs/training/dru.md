# DRU / DIAL training

## Purpose
Specify training using Discretize-Regularize Units (DRU) and Differentiable Inter-Agent Learning (DIAL)
for communication-constrained agents.

## Algorithm (informative)
- Replace hard binary communication with differentiable proxies
- Regularize outputs toward discrete values
- Anneal regularization during training

## Constraints
- Runtime communication MUST remain discrete
- DRU/DIAL MUST NOT increase communication bandwidth
- Learned policies MUST satisfy all player contracts

## Notes
DRU/DIAL is optional and experimental in QSeaBattle.