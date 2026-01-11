# Class LinMeasurementLayerA

**Module import path**: `Q_Sea_Battle.lin_measurement_layer_a.LinMeasurementLayerA`

> Learnable mapping from a flattened field vector to per-cell measurement probabilities in `[0, 1]`.

!!! note "Parent class"
    Inherits from `tf.keras.layers.Layer`.

!!! note "Derived symbols"
    Let `field_size = n`, `n2 = n**2`, and `comms_size = m`.

## Overview

Maps an input field vector of length `n2` to per-cell probabilities.

## call

- Input: `tf.Tensor`, shape `(n2,)` or `(B, n2)`
- Output: `tf.Tensor`, shape `(n2,)` or `(B, n2)`, values in `[0, 1]`

## Changelog

- 2026-01-11 — Author: Technical Documentation Team
