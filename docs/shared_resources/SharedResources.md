# Shared Resources

This page documents the **types of shared resources** used in QSeaBattle and clarifies their conceptual meaning and implementation scope.



## Overview

A *shared resource* is any pre-established correlation available to both players before the game starts and usable during play without communication. Different resource classes define different power levels for player strategies.

The hierarchy below is ordered from weakest to strongest.

## No shared resource (Local / deterministic)

**Resource**
None.

**Meaning**
Players act independently. Outputs are deterministic (or equivalently locally random) functions of their own inputs.

**Status in QSeaBattle**

* Baseline classical strategies.
* Used implicitly when no shared resource object is present.

## Shared classical randomness (SR)

**Resource**
A shared classical random variable (common seed).

**Meaning**
Players correlate their actions via pre-agreed randomness, but correlations are entirely classical and local.

**Key property**

* Does **not** violate Bell inequalities.
* Optimal strategies can be assumed deterministic given the shared seed.

**Status in QSeaBattle**

* Not yet implemented.

## Entanglement-assisted resources (EA)

**Resource**
A shared quantum state with local measurements.

**Meaning**
Players exploit quantum correlations that exceed classical limits but respect Tsirelson bounds.

**Key property**

* No signalling.
* Correlations achievable by quantum mechanics.

**Status in QSeaBattle**

* Not yet implemented.

## PR-assisted resources (PR)

**Resource**
A classical **no-signalling box** with correlations stronger than classical and tunable via a parameter.

**Meaning**
Models *post-quantum* correlations (PR-box–like) while remaining classical and efficiently simulable.

**Key property**

* No signalling.
* Can exceed classical and quantum bounds depending on configuration.
* Strictly stronger than shared randomness.

**Status in QSeaBattle**

* Implemented by `PRAssisted`.
* Used by:

  * `PRAssistedPlayers`
  * `PRAssistedPlayerA`
  * `PRAssistedPlayerB`

## Summary table (conceptual)

* **Local**: no correlation
* **SR**: classical correlation only (deprecated)
* **EA**: quantum correlation (reference)
* **PR**: post-quantum, no-signalling correlation (active model)

## Changelog

* 2026-01-10 — Author: Rob Hendriks — Initial documentation of shared resource classes and terminology.
