"""Tests for the Tournament class."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")

import numpy as np

from Q_Sea_Battle.game_env import GameEnv
from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.players_base import PlayerA, PlayerB, Players
from Q_Sea_Battle.tournament import Tournament


class DummyPlayerA(PlayerA):
    """Deterministic PlayerA used for testing Tournament logging."""

    def __init__(self, layout: GameLayout) -> None:
        super().__init__(layout)
        self._last_logprob = 0.0

    def decide(self, field: np.ndarray, supp=None) -> np.ndarray:  # type: ignore[override]
        # Always send all zeros as communication.
        m = self.game_layout.comms_size
        comm = np.zeros(m, dtype=int)
        self._last_logprob = 0.0
        return comm

    def get_log_prob(self) -> float:
        return self._last_logprob

    def get_prev(self):
        # Minimal previous info: two small lists.
        return (["meas"], ["out"])


class DummyPlayerB(PlayerB):
    """Deterministic PlayerB used for testing Tournament logging."""

    def __init__(self, layout: GameLayout) -> None:
        super().__init__(layout)
        self._last_logprob = 0.0

    def decide(self, gun: np.ndarray, comm: np.ndarray, supp=None) -> int:  # type: ignore[override]
        # Always don't shoot.
        self._last_logprob = 0.0
        return 0

    def get_log_prob(self) -> float:
        return self._last_logprob


class DummyPlayers(Players):
    """Players factory exposing log_probs and prev for Tournament tests."""

    has_log_probs: bool = True
    has_prev: bool = True

    def __init__(self, layout: GameLayout) -> None:
        super().__init__(layout)
        self._player_a = DummyPlayerA(self.game_layout)
        self._player_b = DummyPlayerB(self.game_layout)

    def players(self):
        # Return the same instances each time.
        return self._player_a, self._player_b


def test_tournament_runs_and_logs_expected_number_of_games() -> None:
    """Tournament.tournament() should produce a log with the right number of rows."""
    layout = GameLayout(field_size=4, comms_size=1, number_of_games_in_tournament=5)
    env = GameEnv(layout)
    players = Players(layout)
    tournament = Tournament(env, players, layout)

    log = tournament.tournament()

    assert len(log.log) == layout.number_of_games_in_tournament
    # Rewards should be 0.0 or 1.0.
    assert set(log.log["reward"].unique()).issubset({0.0, 1.0})


def test_tournament_logs_probs_and_prev_when_available() -> None:
    """Tournament should fill logprob_* and prev_* when players provide them."""
    layout = GameLayout(field_size=4, comms_size=1, number_of_games_in_tournament=3)
    env = GameEnv(layout)
    players = DummyPlayers(layout)
    tournament = Tournament(env, players, layout)

    log = tournament.tournament()

    # There should be one row per game.
    assert len(log.log) == layout.number_of_games_in_tournament

    # logprob_comm and logprob_shoot should not be None.
    assert log.log["logprob_comm"].isna().sum() == 0
    assert log.log["logprob_shoot"].isna().sum() == 0

    # prev_measurements and prev_outcomes should not be None.
    assert log.log["prev_measurements"].isna().sum() == 0
    assert log.log["prev_outcomes"].isna().sum() == 0
