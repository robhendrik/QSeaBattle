import numpy as np
import pytest
import sys
sys.path.append("./src")

class _GameStub:
    def __init__(self, game_env, players):
        self.game_env = game_env
        self.players = players
        self.calls = 0

    def play(self):
        # Deterministic single-hot gun at index 0
        n2 = self.game_env.game_layout.field_size ** 2
        m = self.game_env.game_layout.comms_size
        field = np.zeros(n2, dtype=int)
        gun = np.zeros(n2, dtype=int)
        gun[0] = 1
        comm = np.zeros(m, dtype=int)
        shoot = 0
        reward = 1.0
        self.calls += 1
        return reward, field, gun, comm, shoot


class _PlayersStub:
    def __init__(self, has_log_probs=False, has_prev=False):
        self.has_log_probs = has_log_probs
        self.has_prev = has_prev
        self._pa = self
        self._pb = self

    def players(self):
        return self._pa, self._pb

    def reset(self):
        return None

    def get_log_prob(self):
        return -0.123

    def get_prev(self):
        return (["meas"], ["out"])


@pytest.mark.usefixtures("qsb")
def test_tournament_runs_n_games_and_updates_log(monkeypatch):
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle import tournament as tournament_module

    layout = GameLayout(field_size=4, comms_size=1, number_of_games_in_tournament=3)
    env = GameEnv(layout)
    players = _PlayersStub(has_log_probs=True, has_prev=True)

    # Patch Game used inside tournament module
    monkeypatch.setattr(tournament_module, "Game", _GameStub)

    t = tournament_module.Tournament(game_env=env, players=players, game_layout=layout)
    log = t.tournament()

    assert len(log.log) == 3
    # Check indicators were written for the last row at least
    assert log.log.iloc[-1]["game_id"] == 2
    assert log.log.iloc[-1]["tournament_id"] == 0
    assert log.log.iloc[-1]["meta_id"] == 0

    # Optional hooks should have been recorded
    assert log.log.iloc[-1]["logprob_comm"] is not None
    assert log.log.iloc[-1]["prev_measurements"] is not None
