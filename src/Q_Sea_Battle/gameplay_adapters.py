"""
gameplay_adapters_purelogit.py

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
    Adapter for gameplay Model A when the internal model is pure-logit.

    Internal model A must implement:
        comm_logits, meas_list, out_list = internal_model_a.compute_with_internal(field_logits, harden_between_levels=..., beta_for_hardening=...)

    where field_logits is float32 logits (B,n2) and all returned tensors are logits.

    This adapter converts:
      - boundary input bits -> internal logits (via hard_logit)
      - internal logits -> boundary bits (via sign thresholding), with optional exploration noise on comm logits.
    """
    internal_model_a: Any
    beta: float = 10.0
    harden_between_levels: bool = False  # whether to harden logits between levels to reflect deterministic measurements/outcomes in gameplay mode   
    
    def __post_init__(self):
        self.assert_inputs_binary = True  # Player A must provide bits at the boundary, which we convert to logits

    def compute_with_internal(self, field_batch: tf.Tensor):
        print("Warning: [GameplayModelAAdapter] compute_with_internal called. Method compute_with_internal is deprecated; please call the adapter instance directly to invoke the internal model, e.g. comm_bits, meas_list_bits, out_list_bits = model_a(field_batch)")
        return self(field_batch)
    
    def __call__(
        self,
        field_batch: tf.Tensor,
        explore: bool = False,
        return_comm_logits: bool = False
    ) -> Any:
        field_bits = _ensure_rank2(field_batch, "field_batch")
        if self.assert_inputs_binary:
            _assert_binary_tensor(field_bits, "field_batch")

        field_logits = hard_logit(field_bits, self.beta)

        comm_logits, meas_list, out_list = self.internal_model_a.compute_with_internal(
            field_logits,
            harden_between_levels=self.harden_between_levels,
            beta_for_hardening=self.beta)

        comm_logits = _ensure_rank2(comm_logits, "comm_logits")

        meas_list_logits = _ensure_list_of_rank2(meas_list, "meas_list")
        out_list_logits = _ensure_list_of_rank2(out_list, "out_list")

        meas_list_bits = [tf.where(t >= 0.0, tf.ones_like(t), tf.zeros_like(t)) for t in meas_list_logits]
        out_list_bits = [tf.where(t >= 0.0, tf.ones_like(t), tf.zeros_like(t)) for t in out_list_logits]
        
        if explore:
            comm_logits_eff = comm_logits + tf.random.normal(tf.shape(comm_logits), stddev=0.5)
        else:
            comm_logits_eff = comm_logits

        comm_bits = tf.cast(comm_logits_eff >= 0.0, tf.float32)

        if return_comm_logits:
            return comm_bits, meas_list_bits, out_list_bits, comm_logits
        else:
            return comm_bits, meas_list_bits, out_list_bits


@dataclass
class GameplayModelBAdapter:
    """
    Adapter for gameplay Model B when the internal model is pure-logit.

    Player B provides gun, comm and prev_* as *bits*  and the adapter converts them to logits before passing to internal model B. Specifically:
    - gun_bits  -> gun_logits  via hard_logit(beta)
    - comm_bits -> comm_logits via hard_logit(beta)
    - prev_meas / prev_out bits -> logits via hard_logit (beta)

    Internal model B must implement:
        shoot_logit, *_ = internal_model_b.compute_with_internal(gun_logits, comm_logits, prev
    The internal model B provides shoot_logit as output, 
    - shoot_logit -> shoot bits via (shoot_logit >= 0) thresholding.
    
    """
    internal_model_b: Any
    beta: float = 10.0
    harden_between_levels: bool = False  # whether to harden logits between levels to reflect deterministic measurements/outcomes in gameplay mode   

    def __post_init__(self):
        self.assert_inputs_binary_gun_comm = True
        self.assert_inputs_binary_prev_meas_out = True
        if hasattr(self.internal_model_b, "harden_between_levels"):
            self.internal_model_b.harden_between_levels = True

    def compute_with_internal(self, inputs: Sequence[Any]) -> tf.Tensor:
        print("Warning: [GameplayModelBAdapter] compute_with_internal called. Method compute_with_internal is deprecated; please call the adapter instance directly to invoke the internal model, e.g. shoot_bit = model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])")
        return self(inputs)
    
    def __call__(self, inputs: Sequence[Any], explore: bool = False, return_shoot_logit: bool = False) -> tf.Tensor:
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
        
        shoot_logit, *_ = self.internal_model_b.compute_with_internal(gun_logits, 
                                                                  comm_logits, 
                                                                  prev_meas_logits, 
                                                                  prev_out_logits,
                                                                  harden_between_levels=self.harden_between_levels,
                                                                  beta_for_hardening=self.beta)
        shoot_logit = _ensure_rank2(shoot_logit, "shoot_logit")

        if explore:
            shoot_logit_eff = shoot_logit + tf.random.normal(tf.shape(shoot_logit), stddev=0.5)
        else:
            shoot_logit_eff = shoot_logit

        shoot_bit = tf.cast(shoot_logit_eff >= 0.0, tf.float32)

        if return_shoot_logit:
            return shoot_bit, shoot_logit
        else:
            return shoot_bit
