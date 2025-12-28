"""Tests for NeuralNetPlayers factory and training (v0.3)."""

import sys

import numpy as np
import pandas as pd

sys.path.append("./src")

import Q_Sea_Battle as qsb  # type: ignore[import]


def test_neural_net_players_creation_and_explore_flag() -> None:
    """NeuralNetPlayers should create consistent A/B players and propagate explore."""
    layout = qsb.GameLayout(field_size=2, comms_size=1)
    players = qsb.NeuralNetPlayers(layout)

    # Initial explore flag is False.
    player_a, player_b = players.players()
    assert isinstance(player_a, qsb.NeuralNetPlayerA)
    assert isinstance(player_b, qsb.NeuralNetPlayerB)
    assert player_a.explore is False
    assert player_b.explore is False

    # Change explore and check it propagates.
    players.set_explore(True)
    player_a2, player_b2 = players.players()
    assert player_a2.explore is True
    assert player_b2.explore is True
    # Same instances are reused.
    assert player_a2 is player_a
    assert player_b2 is player_b


def _build_simple_supervised_dataset(
    layout: qsb.GameLayout, num_samples: int = 8
) -> pd.DataFrame:
    """Helper to build a tiny synthetic supervised dataset.

    The semantics are deliberately simple; the goal is just to ensure that
    training runs end-to-end and the players remain usable.
    """
    n2 = layout.field_size ** 2
    m = layout.comms_size

    fields: list[np.ndarray] = []
    guns: list[np.ndarray] = []
    comms: list[np.ndarray] = []
    shoots: list[int] = []

    rng = np.random.default_rng(123)

    for _ in range(num_samples):
        field = rng.integers(0, 2, size=n2, dtype=int)
        gun = np.zeros(n2, dtype=int)
        gun[rng.integers(0, n2)] = 1
        # Simple communication heuristic: repeat the cell value (or majority)
        # in all bits so that a non-trivial training signal exists.
        cell_value = int(field[gun == 1][0])
        comm = np.full(m, cell_value, dtype=int)
        shoot = cell_value

        fields.append(field)
        guns.append(gun)
        comms.append(comm)
        shoots.append(shoot)

    df = pd.DataFrame(
        {
            "field": fields,
            "gun": guns,
            "comm": comms,
            "shoot": shoots,
        }
    )
    return df


def test_neural_net_players_train_model_a_and_b() -> None:
    """Training via train_model_a / train_model_b should run and players work."""
    layout = qsb.GameLayout(field_size=2, comms_size=1)
    players = qsb.NeuralNetPlayers(layout)

    df = _build_simple_supervised_dataset(layout, num_samples=32)

    training_settings = {"epochs": 1, "batch_size": 8, "verbose": 0}

    # Train A on field -> comm.
    players.train_model_a(df, training_settings)

    # For B we typically want student comms; for this smoke test we just
    # reuse the teacher comms already present in the DataFrame.
    players.train_model_b(df, training_settings)

    # After training, models should be available and players can act.
    player_a, player_b = players.players()

    n2 = layout.field_size ** 2
    m = layout.comms_size

    field = np.array([0, 1, 1, 0], dtype=int).reshape(n2)
    gun = np.array([0, 0, 1, 0], dtype=int).reshape(n2)

    comm = player_a.decide(field)
    assert comm.shape == (m,)

    shoot = player_b.decide(gun, comm)
    assert shoot in (0, 1)


def test_neural_net_players_train_legacy_noop() -> None:
    """Legacy train() should be a harmless no-op that leaves players usable."""
    layout = qsb.GameLayout(field_size=2, comms_size=1)
    players = qsb.NeuralNetPlayers(layout)

    df = _build_simple_supervised_dataset(layout, num_samples=8)

    # Calling the legacy method should not raise.
    players.train(df, {"epochs": 1, "batch_size": 4, "verbose": 0})

    # Players can still be constructed and used with randomly initialised
    # models.
    player_a, player_b = players.players()

    n2 = layout.field_size ** 2
    m = layout.comms_size

    field = np.array([1, 0, 1, 0], dtype=int).reshape(n2)
    gun = np.array([0, 1, 0, 0], dtype=int).reshape(n2)

    comm = player_a.decide(field)
    assert comm.shape == (m,)

    shoot = player_b.decide(gun, comm)
    assert shoot in (0, 1)
