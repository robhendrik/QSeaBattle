"""Neural-network player factory, persistence, and training utilities.

This module provides :class:`NeuralNetPlayers`, a :class:`~.players_base.Players`
implementation that constructs and owns a matched pair of neural players:

* :class:`~.neural_net_player_a.NeuralNetPlayerA` (Player A), which produces a
  communication vector as logits given the current field.
* :class:`~.neural_net_player_b.NeuralNetPlayerB` (Player B), which decides
  whether to shoot (single logit) given the communicated logits and a gun
  position encoding.

The factory centralizes model construction and (de)serialization so that all
player instances share the same underlying Keras models.

Notes:
    This module enables eager execution for TensorFlow functions via
    ``tf.config.run_functions_eagerly(True)`` to simplify debugging and to make
    per-step behavior easier to inspect.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.run_functions_eagerly(True)

from .game_layout import GameLayout
from .players_base import Players, PlayerA, PlayerB
from .neural_net_player_a import NeuralNetPlayerA, _scale_field
from .neural_net_player_b import NeuralNetPlayerB, _gun_one_hot_to_index


class NeuralNetPlayers(Players):
    """Factory for neural-network-based Player A and Player B.

    The factory owns a pair of Keras models and hands references to them to the
    created players:

    * ``model_a``: maps a scaled, flattened field (length ``n2``) to a
      communication vector of logits (length ``m``).
    * ``model_b``: maps a compact representation consisting of the normalized
      gun index (scalar) concatenated with the communication logits (length
      ``m``) to a single shoot logit.

    Logits are used throughout the internal model interfaces. Logical bit values
    represented as logits are determined by the logit sign.

    The legacy :meth:`train` method is retained for backward compatibility but
    is currently a no-op and emits a warning. Use :meth:`train_model_a` and
    :meth:`train_model_b` instead.
    """

    #: Whether Tournament should attempt to read log-probabilities via
    #: ``get_log_prob`` from the underlying players.
    has_log_probs: bool = True

    def __init__(
        self,
        game_layout: Optional[GameLayout] = None,
        model_a: Optional[tf.keras.Model] = None,
        model_b: Optional[tf.keras.Model] = None,
        explore: bool = False,
    ) -> None:
        """Initialize a :class:`NeuralNetPlayers` factory.

        Args:
            game_layout: Optional :class:`~.game_layout.GameLayout`. If ``None``,
                a default layout is constructed.
            model_a: Optional pre-constructed communication model for Player A.
                If ``None``, a default architecture is created on first use.
            model_b: Optional pre-constructed shoot model for Player B.
                If ``None``, a default architecture is created on first use.
            explore: Exploration flag propagated to created players.
        """
        if game_layout is None:
            game_layout = GameLayout()  # type: ignore[call-arg]
        super().__init__(game_layout=game_layout)

        self.game_layout: GameLayout = game_layout
        self.explore: bool = explore

        self.model_a: Optional[tf.keras.Model] = model_a
        self.model_b: Optional[tf.keras.Model] = model_b

        self._playerA: Optional[NeuralNetPlayerA] = None
        self._playerB: Optional[NeuralNetPlayerB] = None

    # ------------------------------------------------------------------
    # Player factory interface
    # ------------------------------------------------------------------
    def players(self) -> Tuple[PlayerA, PlayerB]:
        """Create or return a neural Player A/B pair.

        Players are created lazily and cached. If models are not provided at
        construction time, default architectures are built based on the current
        :class:`~.game_layout.GameLayout`.

        Returns:
            A tuple ``(player_a, player_b)``.
        """
        if self.model_a is None:
            self.model_a = self._build_model_a()
        if self.model_b is None:
            self.model_b = self._build_model_b()

        if self._playerA is None:
            self._playerA = NeuralNetPlayerA(
                game_layout=self.game_layout,
                model_a=self.model_a,
                explore=self.explore,
            )
        if self._playerB is None:
            self._playerB = NeuralNetPlayerB(
                game_layout=self.game_layout,
                model_b=self.model_b,
                explore=self.explore,
            )

        return self._playerA, self._playerB

    def reset(self) -> None:
        """Reset per-game state of the created players.

        This clears per-game state such as accumulated log-probabilities, but
        does not modify the underlying Keras model parameters.
        """
        if self._playerA is not None:
            self._playerA.reset()
        if self._playerB is not None:
            self._playerB.reset()

    def set_explore(self, flag: bool) -> None:
        """Set exploration behavior for both created players.

        Args:
            flag: If ``True``, players act stochastically. If ``False``, players
                act deterministically (e.g., by thresholding probabilities).
        """
        self.explore = flag
        if self._playerA is not None:
            self._playerA.explore = flag
        if self._playerB is not None:
            self._playerB.explore = flag

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def store_models(self, filenameA: str, filenameB: str) -> None:
        """Serialize the underlying Keras models to disk.

        Args:
            filenameA: Output path for Player A's model.
            filenameB: Output path for Player B's model.
        """
        if self.model_a is None:
            self.model_a = self._build_model_a()
        if self.model_b is None:
            self.model_b = self._build_model_b()

        self.model_a.save(filenameA)
        self.model_b.save(filenameB)

    def load_models(self, filenameA: str, filenameB: str) -> None:
        """Load Keras models from disk and attach them to this factory.

        Any already-created players are updated in-place to reference the newly
        loaded models.

        Args:
            filenameA: Path to a serialized Player A model.
            filenameB: Path to a serialized Player B model.
        """
        self.model_a = tf.keras.models.load_model(filenameA)
        self.model_b = tf.keras.models.load_model(filenameB)

        if self._playerA is not None and self.model_a is not None:
            self._playerA.model_a = self.model_a
        if self._playerB is not None and self.model_b is not None:
            self._playerB.model_b = self.model_b

    # ------------------------------------------------------------------
    # Legacy training API
    # ------------------------------------------------------------------
    def train(self, dataset, training_settings):  # type: ignore[override]
        """Legacy training API (no-op).

        Historically, this method trained both models jointly from a single
        dataset. The recommended interface is now :meth:`train_model_a` and
        :meth:`train_model_b`, which separate data requirements and match the
        current imitation-learning workflow.

        For backward compatibility, this method emits a warning and returns
        without performing training.
        """
        warnings.warn(
            "NeuralNetPlayers.train() is deprecated. Use train_model_a() and "
            "train_model_b() instead.",
            UserWarning,
            stacklevel=2,
        )
        return

    # ------------------------------------------------------------------
    # New training APIs
    # ------------------------------------------------------------------
    def train_model_a(self, dataset, training_settings):
        """Train the communication model (``model_a``).

        Expected dataset columns:
            - ``field``: per-sample field array.
            - ``comm``: per-sample teacher communication vector.
            - ``sample_weight`` (optional): per-sample weight if enabled.

        The model is trained with binary cross-entropy from logits.

        Args:
            dataset: Pandas DataFrame-like object providing the required columns.
            training_settings: Mapping of training hyperparameters. Supported
                keys include ``epochs``, ``batch_size``, ``learning_rate``,
                ``verbose``, and ``use_sample_weight``.
        """
        if self.model_a is None:
            self.model_a = self._build_model_a()

        layout = self.game_layout
        n2 = layout.field_size ** 2
        m = layout.comms_size

        fields = np.stack(dataset["field"].to_numpy(), axis=0).astype("float32")
        fields = fields.reshape((-1, n2))
        fields_scaled = _scale_field(fields)

        comms_teacher = np.stack(dataset["comm"].to_numpy(), axis=0).astype("float32")
        comms_teacher = comms_teacher.reshape((-1, m))

        use_sample_weight = bool(training_settings.get("use_sample_weight", False))
        if use_sample_weight and "sample_weight" in dataset.columns:
            sample_weight = (
                dataset["sample_weight"].to_numpy().astype("float32").reshape((-1,))
            )
        else:
            sample_weight = None

        epochs = int(training_settings.get("epochs", 3))
        batch_size = int(training_settings.get("batch_size", 32))
        learning_rate = float(training_settings.get("learning_rate", 1e-3))
        verbose = int(training_settings.get("verbose", 0))

        opt_a = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model_a.compile(
            optimizer=opt_a,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self.model_a.fit(
            fields_scaled,
            comms_teacher,
            sample_weight=sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def train_model_b(self, dataset, training_settings):
        """Train the shoot model (``model_b``).

        Expected dataset columns:
            - ``gun``: one-hot gun position over the flattened field.
            - ``comm``: communication logits or targets (length ``m``).
            - ``shoot``: binary shoot label (shape ``(N, 1)`` after reshape).
            - ``sample_weight`` (optional): per-sample weight if enabled.

        The gun one-hot vector is converted to a normalized scalar index via
        :func:`~.neural_net_player_b._gun_one_hot_to_index` and concatenated
        with the communication vector before training.

        Args:
            dataset: Pandas DataFrame-like object providing the required columns.
            training_settings: Mapping of training hyperparameters. Supported
                keys include ``epochs``, ``batch_size``, ``learning_rate``,
                ``verbose``, and ``use_sample_weight``.
        """
        if self.model_b is None:
            self.model_b = self._build_model_b()

        layout = self.game_layout
        n2 = layout.field_size ** 2
        m = layout.comms_size

        guns = np.stack(dataset["gun"].to_numpy(), axis=0).astype("float32")
        guns = guns.reshape((-1, n2))
        gun_idx_norm = _gun_one_hot_to_index(guns)  # shape (N, 1)

        comms = np.stack(dataset["comm"].to_numpy(), axis=0).astype("float32")
        comms = comms.reshape((-1, m))

        x = np.concatenate([gun_idx_norm, comms], axis=1)

        shoots = dataset["shoot"].to_numpy().astype("float32").reshape((-1, 1))

        use_sample_weight = bool(training_settings.get("use_sample_weight", False))
        if use_sample_weight and "sample_weight" in dataset.columns:
            sample_weight = (
                dataset["sample_weight"].to_numpy().astype("float32").reshape((-1,))
            )
        else:
            sample_weight = None

        epochs = int(training_settings.get("epochs", 3))
        batch_size = int(training_settings.get("batch_size", 32))
        learning_rate = float(training_settings.get("learning_rate", 1e-3))
        verbose = int(training_settings.get("verbose", 0))

        opt_b = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model_b.compile(
            optimizer=opt_b,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self.model_b.fit(
            x,
            shoots,
            sample_weight=sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Internal model builders
    # ------------------------------------------------------------------
    def _build_model_a(self) -> tf.keras.Model:
        """Build the default communication model for Player A.

        Returns:
            A Keras model that maps ``field_scaled`` (shape ``(n2,)``) to
            communication logits (shape ``(m,)``).
        """
        n2 = self.game_layout.field_size ** 2
        m = self.game_layout.comms_size

        inputs = tf.keras.Input(shape=(n2,), name="field_scaled")

        hidden_dim = max(32, min(256, n2))
        x = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            name="a_dense_0",
        )(inputs)
        x = tf.keras.layers.Dense(
            max(m * 2, 16),
            activation="relu",
            name="a_dense_1",
        )(x)

        outputs = tf.keras.layers.Dense(
            m,
            activation=None,
            name="comm_logits",
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="neural_net_A_v3")
        return model

    def _build_model_b(self) -> tf.keras.Model:
        """Build the default shoot model for Player B.

        Returns:
            A Keras model that maps ``gunidx_comm`` (shape ``(1 + m,)``) to a
            single shoot logit (shape ``(1,)``).
        """
        m = self.game_layout.comms_size
        in_dim = 1 + m

        inputs = tf.keras.Input(shape=(in_dim,), name="gunidx_comm")

        hidden_dim = max(16, m * 4)
        x = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            name="b_dense_0",
        )(inputs)
        x = tf.keras.layers.Dense(
            max(8, m * 2),
            activation="relu",
            name="b_dense_1",
        )(x)

        outputs = tf.keras.layers.Dense(
            1,
            activation=None,
            name="shoot_logit",
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="neural_net_B_v3")
        return model