import numpy as np
import pytest
import sys
sys.path.append("./src")

class _StubPlayerA:
    def __init__(self, comm):
        self._comm = np.asarray(comm, dtype=int)
        self.calls = []

    def decide(self, field, supp=None):
        self.calls.append(("decide", np.asarray(field).shape, supp))
        return self._comm.copy()


class _StubPlayerB:
    def __init__(self, shoot):
        self._shoot = int(shoot)
        self.calls = []

    def decide(self, gun, comm, supp=None):
        self.calls.append(("decide", np.asarray(gun).shape, np.asarray(comm).shape, supp))
        return int(self._shoot)


class _StubPlayers:
    def __init__(self, player_a, player_b):
        self._pa = player_a
        self._pb = player_b
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def players(self):
        return self._pa, self._pb


class _StubEnv:
    def __init__(self, field, gun, noisy_comm=None, reward=1.0):
        self._field = np.asarray(field, dtype=int)
        self._gun = np.asarray(gun, dtype=int)
        self._noisy_comm = noisy_comm
        self._reward = float(reward)
        self.calls = []

    def reset(self):
        self.calls.append("reset")

    def provide(self):
        self.calls.append("provide")
        return self._field.copy(), self._gun.copy()

    def apply_channel_noise(self, comm):
        self.calls.append(("apply_channel_noise", np.asarray(comm).copy()))
        if self._noisy_comm is None:
            return np.asarray(comm, dtype=int).copy()
        return np.asarray(self._noisy_comm, dtype=int).copy()

    def evaluate(self, shoot):
        self.calls.append(("evaluate", int(shoot)))
        return self._reward


@pytest.mark.usefixtures("qsb")
def test_game_play_orchestrates_calls_and_returns_values():
    from Q_Sea_Battle.game import Game

    field = np.zeros(16, dtype=int)
    gun = np.zeros(16, dtype=int)
    gun[3] = 1

    pa = _StubPlayerA(comm=[1, 0])
    pb = _StubPlayerB(shoot=1)
    players = _StubPlayers(pa, pb)

    env = _StubEnv(field=field, gun=gun, noisy_comm=[0, 0], reward=0.0)

    game = Game(game_env=env, players=players)
    reward, out_field, out_gun, out_comm, out_shoot = game.play()

    assert reward == 0.0
    assert np.array_equal(out_field, field)
    assert np.array_equal(out_gun, gun)
    assert np.array_equal(out_comm, np.array([0, 0], dtype=int))
    assert out_shoot in (0, 1)

    # Verify call order and side effects
    assert env.calls[0:2] == ["reset", "provide"]
    assert any(c[0] == "apply_channel_noise" for c in env.calls if isinstance(c, tuple))
    assert env.calls[-1][0] == "evaluate"
    assert players.reset_calls == 1
    assert pa.calls and pb.calls
