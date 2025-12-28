"""Tests for the TournamentLog class."""

from __future__ import annotations

import sys

# Ensure the src folder is on the Python path so Q_Sea_Battle can be imported.
sys.path.append("./src")

import numpy as np
import pandas as pd

from Q_Sea_Battle.game_layout import GameLayout
from Q_Sea_Battle.tournament_log import TournamentLog


def test_update_adds_row_and_preserves_columns() -> None:
    """update() should append a row with all required columns."""
    layout = GameLayout()
    log = TournamentLog(layout)

    field = np.array([0, 1, 0, 1], dtype=int)
    gun = np.array([0, 1, 0, 0], dtype=int)
    comm = np.array([1], dtype=int)
    shoot = 1
    cell_value = 1
    reward = 1.0

    log.update(field, gun, comm, shoot, cell_value, reward)

    assert len(log.log) == 1
    row = log.log.iloc[0]

    # All expected columns present.
    assert set(layout.log_columns).issubset(set(log.log.columns))

    assert np.array_equal(row["field"], field)
    assert np.array_equal(row["gun"], gun)
    assert np.array_equal(row["comm"], comm)
    assert row["shoot"] == shoot
    assert row["cell_value"] == cell_value
    assert row["reward"] == reward


def test_update_log_probs_updates_last_row() -> None:
    """update_log_probs() should update logprob columns on last row."""
    layout = GameLayout()
    log = TournamentLog(layout)

    field = np.zeros(4, dtype=int)
    gun = np.array([1, 0, 0, 0], dtype=int)
    comm = np.array([0], dtype=int)

    log.update(field, gun, comm, shoot=0, cell_value=0, reward=1.0)
    log.update_log_probs(logprob_comm=-0.5, logprob_shoot=-0.7)

    row = log.log.iloc[0]
    assert row["logprob_comm"] == -0.5
    assert row["logprob_shoot"] == -0.7


def test_update_log_prev_updates_last_row() -> None:
    """update_log_prev() should set prev_measurements and prev_outcomes."""
    layout = GameLayout()
    log = TournamentLog(layout)

    field = np.zeros(4, dtype=int)
    gun = np.array([1, 0, 0, 0], dtype=int)
    comm = np.array([0], dtype=int)

    log.update(field, gun, comm, shoot=0, cell_value=0, reward=1.0)

    prev_meas = ["meas"]
    prev_out = ["out"]
    log.update_log_prev(prev_meas, prev_out)

    row = log.log.iloc[0]
    assert row["prev_measurements"] == prev_meas
    assert row["prev_outcomes"] == prev_out


def test_update_indicators_sets_ids_and_uid() -> None:
    """update_indicators() should set id fields and a non-empty game_uid."""
    layout = GameLayout()
    log = TournamentLog(layout)

    field = np.zeros(4, dtype=int)
    gun = np.array([1, 0, 0, 0], dtype=int)
    comm = np.array([0], dtype=int)

    log.update(field, gun, comm, shoot=0, cell_value=0, reward=1.0)
    log.update_indicators(game_id=5, tournament_id=2, meta_id=99)

    row = log.log.iloc[0]
    assert row["game_id"] == 5
    assert row["tournament_id"] == 2
    assert row["meta_id"] == 99
    assert isinstance(row["game_uid"], str)
    assert len(row["game_uid"]) > 0


def test_outcome_returns_mean_and_std_error() -> None:
    """outcome() should compute mean reward and standard error."""
    layout = GameLayout()
    log = TournamentLog(layout)

    # Create deterministic rewards: [1, 0, 1, 0]
    for val in [1.0, 0.0, 1.0, 0.0]:
        field = np.zeros(4, dtype=int)
        gun = np.array([1, 0, 0, 0], dtype=int)
        comm = np.array([0], dtype=int)
        log.update(field, gun, comm, shoot=0, cell_value=0, reward=val)

    mean_reward, std_error = log.outcome()

    assert mean_reward == 0.5
    # Standard error should be positive for non-constant rewards.
    assert std_error > 0.0
