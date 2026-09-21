"""
Gameplay adapters for a "pure-logit" internal Linear and Pyramid model composition.

In this project, all gameplay-facing inputs/outputs at the player boundary are
binary *bits* (float32 tensors with values in {0.0, 1.0}). Internally, the
composed models operate on *logits* (real-valued float32 tensors), where
the logical bit value is determined by the sign of the logit.

Boundary contracts (project terminology)
----------------------------------------
Player A boundary:
- The Player A-facing model is :class:`GameplayModelAAdapter`.
- Call pattern::

    comm_bits, meas_list_bits, out_list_bits = model_a(field_batch)

- ``field_batch``: float32 bits in {0.0, 1.0}, shape (B, n2)
- Returns:
  - ``comm_bits``: float32 bits in {0.0, 1.0}, shape (B, m)
    Already sampled/decided by the adapter; Player A forwards these.
  - ``meas_list_bits``: Python list of float32 bit tensors, one per level.
  - ``out_list_bits``: Python list of float32 bit tensors, one per level.

Player B boundary:
- The Player B-facing model is :class:`GameplayModelBAdapter`.
- Call pattern::

    shoot_bit = model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])

- ``gun_batch``: float32 bits in {0.0, 1.0}, shape (B, n2), typically one-hot.
- ``comm_batch``: float32 bits in {0.0, 1.0}, shape (B, m), forwarded from A.
- ``prev_meas_batch`` / ``prev_out_batch``: typically list/tuple of bit tensors
  from the previous call to Player A.
- Returns:
  - ``shoot_bit``: float32 bit tensor, shape (B, 1)

Pure-logit translation performed by these adapters
-------------------------------------------------
These adapters implement a "hard-logit" representation for boundary bits:

- Bit -> logit via :func:`hard_logit` with scale ``beta``:
  - b=1 -> +beta
  - b=0 -> -beta
- Logit -> bit via a sign threshold at 0.0:
  - logit >= 0.0 -> 1.0
  - logit < 0.0  -> 0.0

No scaled representation (e.g., ``x - 0.5``) is used in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

import tensorflow as tf


def _as_f32(x: Any) -> tf.Tensor:
    """Convert ``x`` to a ``tf.float32`` tensor."""
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _assert_binary_tensor(x: tf.Tensor, name: str) -> None:
    """Assert that ``x`` contains only binary values {0, 1}.

    This is intended for boundary validation (gameplay bits). In eager mode, the
    assertion is executed immediately; in graph mode it becomes a
    ``tf.debugging`` op.

    Args:
        x: Tensor expected to contain only 0/1 values (integer or float).
        name: Label included in assertion messages.
    """
    x0 = tf.cast(x, tf.float32)
    tf.debugging.assert_greater_equal(x0, 0.0, message=f"{name} must be binary (>=0)")
    tf.debugging.assert_less_equal(x0, 1.0, message=f"{name} must be binary (<=1)")
    tf.debugging.assert_near(x0, tf.round(x0), atol=1e-6, message=f"{name} must be binary (0/1)")


def hard_logit(bits: tf.Tensor, beta: float) -> tf.Tensor:
    """Map bits in {0,1} to hard logits in {-beta,+beta}.

    Args:
        bits: Bit tensor (typically float32) with values in {0, 1}.
        beta: Logit magnitude to use for 0/1.

    Returns:
        A float32 tensor with the same shape as ``bits`` where 0 -> -beta and
        1 -> +beta.
    """
    bits = tf.cast(bits, tf.float32)
    return beta * (2.0 * bits - 1.0)


def _ensure_rank2(x: tf.Tensor, name: str) -> tf.Tensor:
    """Ensure ``x`` is rank-2 with shape (B, D).

    If a rank-1 tensor of shape (D,) is provided, a batch dimension is added,
    producing shape (1, D).

    Args:
        x: Input tensor.
        name: Label used in assertion messages.

    Returns:
        A float32 tensor with rank 2.
    """
    x = _as_f32(x)
    if x.shape.rank == 1:
        x = x[None, :]
    tf.debugging.assert_rank(x, 2, message=f"{name} must be rank-2 (B,D) (or rank-1 D)")
    return x


def _ensure_list_of_rank2(xs: Any, name: str) -> List[tf.Tensor]:
    """Normalize a value into a list of rank-2 tensors.

    Accepts a list/tuple of tensors (preferred) or a single tensor. ``None`` is
    treated as an empty list.

    Args:
        xs: List/tuple of tensors, a single tensor, or ``None``.
        name: Base label used for per-element assertion messages.

    Returns:
        A list of float32 rank-2 tensors.
    """
    if xs is None:
        return []
    if isinstance(xs, (list, tuple)):
        out: List[tf.Tensor] = []
        for i, t in enumerate(xs):
            out.append(_ensure_rank2(t, f"{name}[{i}]"))
        return out
    return [_ensure_rank2(xs, name)]


@dataclass
class GameplayModelAAdapter:
    """Gameplay adapter for Player A with a pure-logit internal model.

    The adapter performs boundary translation:

    - Boundary input: ``field_batch`` bits (float32 in {0, 1}) -> logits using
      :func:`hard_logit` with ``beta``.
    - Internal output: ``comm_logits``, ``meas_list`` logits, ``out_list`` logits
      -> boundary bits via sign thresholding at 0.0.

    The adapter optionally injects exploration noise into ``comm_logits`` before
    thresholding (see ``explore``).

    Expected internal API:
        The internal model must provide::

            comm_logits, meas_list, out_list = internal_model_a.compute_with_internal(
                field_logits,
                harden_between_levels=...,
                beta_for_hardening=...
            )

        where all tensors are logits (float32).

    Attributes:
        internal_model_a: Internal Player A model implementing
            ``compute_with_internal``.
        beta: Magnitude used by :func:`hard_logit` for boundary conversion.
        harden_between_levels: Forwarded to the internal model to optionally
            harden logits between levels for deterministic gameplay behavior.
    """

    internal_model_a: Any
    beta: float = 10.0
    harden_between_levels: bool = False  # Harden logits between levels for deterministic gameplay.

    def __post_init__(self):
        # Gameplay boundaries use bits; enforce this by default.
        self.assert_inputs_binary = True

    def compute_with_internal(self, field_batch: tf.Tensor):
        """Deprecated alias for calling the adapter instance.

        Args:
            field_batch: Bit tensor of shape (B, n2) (or (n2,) which is promoted).

        Returns:
            Same as :meth:`__call__`.

        Note:
            This method prints a deprecation warning and forwards to
            :meth:`__call__`.
        """
        print(
            "Warning: [GameplayModelAAdapter] compute_with_internal called. "
            "Method compute_with_internal is deprecated; please call the adapter instance directly "
            "to invoke the internal model, e.g. comm_bits, meas_list_bits, out_list_bits = model_a(field_batch)"
        )
        return self(field_batch)

    def __call__(
        self,
        field_batch: tf.Tensor,
        explore: bool = False,
        return_comm_logits: bool = False,
    ) -> Any:
        """Run the internal Player A model with bit/logit boundary conversion.

        Args:
            field_batch: Bit tensor (float32 in {0, 1}), shape (B, n2) or (n2,).
            explore: If True, add Gaussian noise (stddev=0.5) to ``comm_logits``
                before thresholding to bits.
            return_comm_logits: If True, also return the raw ``comm_logits`` from
                the internal model (before exploration noise).

        Returns:
            If ``return_comm_logits`` is False:
                Tuple ``(comm_bits, meas_list_bits, out_list_bits)``.
            If ``return_comm_logits`` is True:
                Tuple ``(comm_bits, meas_list_bits, out_list_bits, comm_logits)``.

            Bit tensors are float32 in {0, 1}; list elements are rank-2 tensors.
        """
        field_bits = _ensure_rank2(field_batch, "field_batch")
        if self.assert_inputs_binary:
            _assert_binary_tensor(field_bits, "field_batch")

        field_logits = hard_logit(field_bits, self.beta)

        comm_logits, meas_list, out_list = self.internal_model_a.compute_with_internal(
            field_logits,
            harden_between_levels=self.harden_between_levels,
            beta_for_hardening=self.beta,
        )

        comm_logits = _ensure_rank2(comm_logits, "comm_logits")
        meas_list_logits = _ensure_list_of_rank2(meas_list, "meas_list")
        out_list_logits = _ensure_list_of_rank2(out_list, "out_list")

        # Logit sign encodes the logical bit value at the gameplay boundary.
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
    """Gameplay adapter for Player B with a pure-logit internal model.

    The adapter performs boundary translation:

    - ``gun_bits`` -> ``gun_logits`` via :func:`hard_logit` with ``beta``.
    - ``comm_bits`` -> ``comm_logits`` via :func:`hard_logit` with ``beta``.
    - ``prev_meas_bits`` / ``prev_out_bits`` -> corresponding logits via
      :func:`hard_logit` with ``beta``.
    - Internal output ``shoot_logit`` -> ``shoot_bit`` by thresholding at 0.0.

    Expected internal API:
        The internal model must provide::

            shoot_logit, *_ = internal_model_b.compute_with_internal(
                gun_logits,
                comm_logits,
                prev_meas_logits,
                prev_out_logits,
                harden_between_levels=...,
                beta_for_hardening=...
            )

        where ``shoot_logit`` is a float32 logit tensor.

    Attributes:
        internal_model_b: Internal Player B model implementing
            ``compute_with_internal``.
        beta: Magnitude used by :func:`hard_logit` for boundary conversion.
        harden_between_levels: Forwarded to the internal model to optionally
            harden logits between levels for deterministic gameplay behavior.
    """

    internal_model_b: Any
    beta: float = 10.0
    harden_between_levels: bool = False  # Harden logits between levels for deterministic gameplay.

    def __post_init__(self):
        self.assert_inputs_binary_gun_comm = True
        self.assert_inputs_binary_prev_meas_out = True
        # Some internal models expose a harden_between_levels attribute; force it on for gameplay.
        if hasattr(self.internal_model_b, "harden_between_levels"):
            self.internal_model_b.harden_between_levels = True

    def compute_with_internal(self, inputs: Sequence[Any]) -> tf.Tensor:
        """Deprecated alias for calling the adapter instance.

        Args:
            inputs: Sequence ``[gun_batch, comm_batch, prev_meas_batch, prev_out_batch]``.

        Returns:
            Same as :meth:`__call__`.

        Note:
            This method prints a deprecation warning and forwards to
            :meth:`__call__`.
        """
        print(
            "Warning: [GameplayModelBAdapter] compute_with_internal called. "
            "Method compute_with_internal is deprecated; please call the adapter instance directly "
            "to invoke the internal model, e.g. shoot_bit = model_b([gun_batch, comm_batch, prev_meas_batch, prev_out_batch])"
        )
        return self(inputs)

    def __call__(self, inputs: Sequence[Any], explore: bool = False, return_shoot_logit: bool = False) -> tf.Tensor:
        """Run the internal Player B model with bit/logit boundary conversion.

        Args:
            inputs: Sequence ``[gun_batch, comm_batch, prev_meas_batch, prev_out_batch]``.
            explore: If True, add Gaussian noise (stddev=0.5) to ``shoot_logit``
                before thresholding to a bit.
            return_shoot_logit: If True, return both ``shoot_bit`` and the raw
                ``shoot_logit`` from the internal model (before exploration
                noise).

        Returns:
            If ``return_shoot_logit`` is False:
                ``shoot_bit`` (float32 in {0, 1}), rank-2.
            If ``return_shoot_logit`` is True:
                Tuple ``(shoot_bit, shoot_logit)``.

        Raises:
            ValueError: If ``inputs`` is not a length-4 list/tuple.
        """
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

        shoot_logit, *_ = self.internal_model_b.compute_with_internal(
            gun_logits,
            comm_logits,
            prev_meas_logits,
            prev_out_logits,
            harden_between_levels=self.harden_between_levels,
            beta_for_hardening=self.beta,
        )
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