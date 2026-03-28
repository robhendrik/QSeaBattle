"""
gameplay_adapters.py

Gameplay adapters for a "pure-logit" internal Pyramid model composition.

Context / contracts (as used in this project)
--------------------------------------------
Players operate with *bits* at the gameplay boundary for all signals, but the
internal (composed) models operate purely on *logits*.

Player A boundary:
- For Player A the 'model_a' is the adapter GameplayModelAAdapter
- Player A calls:       comm_bits, meas_list_bits, out_list_bits = model_a.compute_with_internal(field_batch)
- field_batch:          float32 bits in {0.0, 1.0}, shape (B, n2)
- model_a must return:
    * comm_bits:        float32 bits {0.0, 1.0}, shape (B, m)        
        - Already sampled/decided by adapter/model_a; Player A forwards these
        - m: comm width (often 1 in current pyramid dataset, but adapter supports general m).
    * meas_list_bits:   Python list of float32 bits {0.0, 1.0} tensors (one per level)
        - meas_list_bits length = depth 
    * out_list_bits:    Python list of float32 bits {0.0, 1.0} tensors (one per level).
        - out_list_bits length = depth 

Player B boundary:
- For Player B the 'model_b' is the adapter GameplayModelBAdapter
- Player B calls:      shoot_bit = model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])
- gun_batch:           float32 bits in {0.0, 1.0}, shape (B, n2) (one-hot per sample)
- comm_batch:          float32 bits in {0.0, 1.0}, shape (B, m)  (forwarded by Player A)
- prev_meas_batch:     typically list/tuple of tensors (bits) from previous model_a call
- prev_out_batch:      typically list/tuple of tensors (bits) from previous model_a call
- model_b must return:
    * shoot_bit:      float32 bit {0.0, 1.0}, shape (B, 1)

Pure-logit translation performed by these adapters
-------------------------------------------------
Adapter A (for model_a):
- field bits  -> field logits  (hard-logit with beta)
- returns comm_logits/meas_list/out_list logits -> bits via (logit >= 0) thresholding   <-- IMPORTANT

Adapter B (for model_b):
- gun bits    -> gun logits    (hard-logit with beta)
- comm bits   -> comm logits   (hard-logit with beta)   <-- IMPORTANT
- prev_meas / prev_out bits -> logits via hard-logit with beta\
- Adapter B returns shoot logits → bit via thresholding.

No "scaled" representation (x - 0.5) is used anywhere in this module.

Hard-logit mapping
------------------
For a bit b in {0,1}, hard_logit(beta) maps:
- b=1 -> +beta
- b=0 -> -beta

Conversion from logits to bits is done via thresholding at 0.0:
- logit >= 0.0 -> bit=1.0
- logit < 0.0  -> bit=0.0

This keeps magnitudes consistent with the training/converter "hard_logit" representation.

Notes
-----
- The adapters are callable via `__call__` and also expose `compute_with_internal(...)` as a
  backwards-compatible alias that prints a warning.
- Some debug flags (`return_comm_logits`, `return_shoot_logit`) cause adapters to return an
  additional logit tensor alongside the standard bit outputs.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

import tensorflow as tf


def _as_f32(x: Any) -> tf.Tensor:
    """Convert to tf.float32 tensor."""
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _assert_binary_tensor(x: tf.Tensor, name: str) -> None:
    """
    Assert x contains only 0/1 (float or int). Runs in eager mode; in graph mode it becomes a tf.debugging op.
    """
    x0 = tf.cast(x, tf.float32)
    tf.debugging.assert_greater_equal(x0, 0.0, message=f"{name} must be binary (>=0)")
    tf.debugging.assert_less_equal(x0, 1.0, message=f"{name} must be binary (<=1)")
    # also ensure close to integer
    tf.debugging.assert_near(x0, tf.round(x0), atol=1e-6, message=f"{name} must be binary (0/1)")


def hard_logit(bits: tf.Tensor, beta: float) -> tf.Tensor:
    """
    Map bits in {0,1} to hard logits in {-beta,+beta}.
    """
    bits = tf.cast(bits, tf.float32)
    return beta * (2.0 * bits - 1.0)


def _ensure_rank2(x: tf.Tensor, name: str) -> tf.Tensor:
    """
    Ensure tensor is rank-2 (B, D). If rank-1 (D,), add batch dim.
    """
    x = _as_f32(x)
    if x.shape.rank == 1:
        x = x[None, :]
    tf.debugging.assert_rank(x, 2, message=f"{name} must be rank-2 (B,D) (or rank-1 D)")
    return x


def _ensure_list_of_rank2(xs: Any, name: str) -> List[tf.Tensor]:
    """
    Accept list/tuple of tensors (preferred) or a single tensor.
    Returns list of rank-2 tensors.
    """
    if xs is None:
        return []
    if isinstance(xs, (list, tuple)):
        out: List[tf.Tensor] = []
        for i, t in enumerate(xs):
            out.append(_ensure_rank2(t, f"{name}[{i}]"))
        return out
    # fallback: single tensor
    return [_ensure_rank2(xs, name)]

@dataclass
class GameplayModelAAdapter:
    """
    Adapter for gameplay Player A when the internal model is logit-native ("pure-logit").

    Boundary contract (gameplay side):
      - Input `field_batch`: float32 bits in {0.0, 1.0}, shape (B, n2)
      - Output:
          * `comm_bits`: float32 bits in {0.0, 1.0}, shape (B, m)
          * `meas_list_bits`: list of float32 bit tensors (one per level/transition)
          * `out_list_bits`:  list of float32 bit tensors (one per level/transition)

    Internal contract (internal_model_a side):
      - The internal model must implement:
            comm_logits, meas_list_logits, out_list_logits = internal_model_a.compute_with_internal(field_logits)
      - `field_logits` is float32 logits (B, n2). This adapter produces it from bits via `hard_logit(bits, beta)`.
      - `comm_logits`, `meas_list_logits`, `out_list_logits` are logits; this adapter converts them to bits
        by thresholding at 0.0 (logit >= 0 -> 1.0 else 0.0).

    Call style:
      - `__call__` is the primary entry point: `comm_bits, meas_list_bits, out_list_bits = model_a(field_bits)`
      - `compute_with_internal(...)` exists for backwards compatibility and forwards to `__call__`
        (it also prints a warning).

    Exploration:
      - If `explore=True`, Gaussian noise (stddev=0.5) is added to `comm_logits` before thresholding.

    Debug/analysis:
      - If `return_comm_logits=True`, `__call__` returns a 4-tuple:
            (comm_bits, meas_list_bits, out_list_bits, comm_logits)
        Otherwise it returns the standard 3-tuple.
    """
    internal_model_a: Any
    beta: float = 10.0
    explore: bool = False
    return_comm_logits: bool = False  # if True, return logits next of bits (for debugging/analysis)
    
    def __post_init__(self):
        self.assert_inputs_binary = True  # Player A must provide bits at the boundary, which we convert to logits
        if hasattr(self.internal_model_a, "harden_between_levels"):
            self.internal_model_a.harden_between_levels = True  # ensure internal model is robust to hard logits    

    def compute_with_internal(self, field_batch: tf.Tensor):
        print("Warning: [GameplayModelAAdapter] compute_with_internal called. Method compute_with_internal is deprecated; please call the adapter instance directly to invoke the internal model, e.g. comm_bits, meas_list_bits, out_list_bits = model_a(field_batch)")
        return self(field_batch)
    
    def __call__(self, field_batch: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
        field_bits = _ensure_rank2(field_batch, "field_batch")
        if self.assert_inputs_binary:
            _assert_binary_tensor(field_bits, "field_batch")

        field_logits = hard_logit(field_bits, self.beta)

        comm_logits, meas_list, out_list = self.internal_model_a.compute_with_internal(field_logits)

        comm_logits = _ensure_rank2(comm_logits, "comm_logits")

        meas_list_logits = _ensure_list_of_rank2(meas_list, "meas_list")
        out_list_logits = _ensure_list_of_rank2(out_list, "out_list")

        meas_list_bits = [tf.where(t >= 0.0, tf.ones_like(t), tf.zeros_like(t)) for t in meas_list_logits]
        out_list_bits = [tf.where(t >= 0.0, tf.ones_like(t), tf.zeros_like(t)) for t in out_list_logits]
        
        if self.explore:
            comm_logits_eff = comm_logits + tf.random.normal(tf.shape(comm_logits), stddev=0.5)
        else:
            comm_logits_eff = comm_logits

        comm_bits = tf.cast(comm_logits_eff >= 0.0, tf.float32)

        if self.return_comm_logits:
            return comm_bits, meas_list_bits, out_list_bits, comm_logits
        else:
            return comm_bits, meas_list_bits, out_list_bits


@dataclass
class GameplayModelBAdapter:
    """
    Adapter for gameplay Player B when the internal model is logit-native ("pure-logit").

    Boundary contract (gameplay side):
      - Call: `shoot_bit = model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])`
      - `gun_batch`:  float32 bits in {0.0, 1.0}, shape (B, n2) (typically one-hot)
      - `comm_batch`: float32 bits in {0.0, 1.0}, shape (B, m)
      - `prev_meas_batch`, `prev_out_batch`: usually lists/tuples of bit tensors returned by Player A in
        the previous step (may be empty).

    Internal contract (internal_model_b side):
      - Inputs are converted to logits via `hard_logit(bits, beta)`:
          * gun_bits  -> gun_logits
          * comm_bits -> comm_logits
          * prev_meas/out bits -> prev_meas/out logits (per tensor)
      - The internal model is then invoked as:
            shoot_logit = internal_model_b([gun_logits, comm_logits, prev_meas_logits, prev_out_logits])
      - `shoot_logit` is converted to a gameplay bit by thresholding at 0.0:
            shoot_bit = 1.0 if shoot_logit >= 0.0 else 0.0

    Call style:
      - `__call__` is the primary entry point.
      - `compute_with_internal(...)` exists for backwards compatibility and forwards to `__call__`
        (it also prints a warning).

    Exploration:
      - If `explore=True`, Gaussian noise (stddev=0.5) is added to `shoot_logit` before thresholding.

    Debug/analysis:
      - If `return_shoot_logit=True`, `__call__` returns a tuple:
            (shoot_bit, shoot_logit)
        Otherwise it returns only `shoot_bit`.
    """
    internal_model_b: Any
    beta: float = 10.0
    explore: bool = False
    return_shoot_logit: bool = False  # if True, return shoot_logit next to shoot_bit (for debugging/analysis)

    def __post_init__(self):
        self.assert_inputs_binary_gun_comm = True
        self.assert_inputs_binary_prev_meas_out = True
        if hasattr(self.internal_model_b, "harden_between_levels"):
            self.internal_model_b.harden_between_levels = True

    def compute_with_internal(self, inputs: Sequence[Any]) -> tf.Tensor:
        print("Warning: [GameplayModelBAdapter] compute_with_internal called. Method compute_with_internal is deprecated; please call the adapter instance directly to invoke the internal model, e.g. shoot_bit = model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])")
        return self(inputs)
    
    def __call__(self, inputs: Sequence[Any]) -> tf.Tensor:
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 4:
            raise ValueError("model_b adapter expects inputs=[gun_batch, comm_batch, prev_meas_batch, prev_out_batch]")

        gun_batch, comm_batch, prev_meas_batch, prev_out_batch = inputs

        gun_bits = _ensure_rank2(gun_batch, "gun_batch")
        comm_bits = _ensure_rank2(comm_batch, "comm_batch")

        if self.assert_inputs_binary_gun_comm:
            _assert_binary_tensor(gun_bits, "gun_batch")
            _assert_binary_tensor(comm_bits, "comm_batch")

        gun_logits = hard_logit(gun_bits, self.beta)
        comm_logits = hard_logit(comm_bits, self.beta)

        prev_meas_bits = _ensure_list_of_rank2(prev_meas_batch, "prev_meas_batch")
        prev_out_bits = _ensure_list_of_rank2(prev_out_batch, "prev_out_batch")

        if self.assert_inputs_binary_prev_meas_out:
            for i, t in enumerate(prev_meas_bits):
                _assert_binary_tensor(t, f"prev_meas_batch[{i}]")
            for i, t in enumerate(prev_out_bits):
                _assert_binary_tensor(t, f"prev_out_batch[{i}]")

        prev_meas_logits = [hard_logit(t, self.beta) for t in prev_meas_bits]
        prev_out_logits = [hard_logit(t, self.beta) for t in prev_out_bits]
        
        shoot_logit = self.internal_model_b([gun_logits, comm_logits, prev_meas_logits, prev_out_logits])
        shoot_logit = _ensure_rank2(shoot_logit, "shoot_logit")

        if self.explore:
            shoot_logit_eff = shoot_logit + tf.random.normal(tf.shape(shoot_logit), stddev=0.5)
        else:
            shoot_logit_eff = shoot_logit

        shoot_bit = tf.cast(shoot_logit_eff >= 0.0, tf.float32)

        if self.return_shoot_logit:
            return shoot_bit, shoot_logit
        else:
            return shoot_bit
