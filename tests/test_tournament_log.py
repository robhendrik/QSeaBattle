import numpy as np
import pytest
import sys
sys.path.append("./src")

@pytest.mark.usefixtures("qsb")
def test_tournament_log_update_and_outcome():
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.tournament_log import TournamentLog

    layout = GameLayout(field_size=4, comms_size=1)
    log = TournamentLog(layout)
    assert list(log.log.columns) == layout.log_columns
    assert len(log.log) == 0

    field = np.zeros(layout.field_size ** 2, dtype=int)
    gun = np.zeros(layout.field_size ** 2, dtype=int)
    gun[0] = 1
    comm = np.array([1], dtype=int)

    log.update(field=field, gun=gun, comm=comm, shoot=0, cell_value=0, reward=1.0)
    assert len(log.log) == 1

    mean, se = log.outcome()
    assert mean == 1.0
    assert se == 0.0

    log.update(field=field, gun=gun, comm=comm, shoot=1, cell_value=0, reward=0.0)
    mean2, se2 = log.outcome()
    assert mean2 == 0.5
    assert se2 >= 0.0


@pytest.mark.usefixtures("qsb")
def test_tournament_log_update_requires_nonempty_for_last_row_ops():
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.tournament_log import TournamentLog

    log = TournamentLog(GameLayout())
    with pytest.raises(RuntimeError):
        log.update_log_probs(0.0, 0.0)
    with pytest.raises(RuntimeError):
        log.update_log_prev([], [])
    with pytest.raises(RuntimeError):
        log.update_indicators(0, 0, 0)


@pytest.mark.usefixtures("qsb")
def test_tournament_log_last_row_updates():
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.tournament_log import TournamentLog

    layout = GameLayout(field_size=4, comms_size=1)
    log = TournamentLog(layout)

    field = np.zeros(layout.field_size ** 2, dtype=int)
    gun = np.zeros(layout.field_size ** 2, dtype=int)
    gun[1] = 1
    comm = np.array([0], dtype=int)

    log.update(field=field, gun=gun, comm=comm, shoot=0, cell_value=0, reward=1.0)
    log.update_log_probs(-0.1, -0.2)
    log.update_log_prev(prev_meas=[1, 2], prev_out=[0, 1])
    log.update_indicators(game_id=7, tournament_id=0, meta_id=0)

    row = log.log.iloc[-1]
    assert float(row["logprob_comm"]) == pytest.approx(-0.1)
    assert float(row["logprob_shoot"]) == pytest.approx(-0.2)
    assert row["game_id"] == 7
    assert isinstance(row["game_uid"], str) and len(row["game_uid"]) > 0
    assert row["prev_measurements"] == [1, 2]
    assert row["prev_outcomes"] == [0, 1]
