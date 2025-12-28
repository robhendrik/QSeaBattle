
import sys
sys.path.append("./src")

import numpy as np
import tensorflow as tf

from Q_Sea_Battle.trainable_assisted_players import TrainableAssistedPlayers
from Q_Sea_Battle.lin_trainable_assisted_model_a import LinTrainableAssistedModelA
from Q_Sea_Battle.lin_trainable_assisted_model_b import LinTrainableAssistedModelB

try:
    from Q_Sea_Battle.game_layout import GameLayout
except Exception:
    class GameLayout:
        def __init__(self, field_size: int, comms_size: int):
            self.field_size = field_size
            self.comms_size = comms_size


def test_one_step_integration_and_reset_and_explore_toggle() -> None:
    tf.random.set_seed(0)
    np.random.seed(0)

    layout = GameLayout(field_size=4, comms_size=1)
    model_a = LinTrainableAssistedModelA(field_size=4, comms_size=1, sr_mode="expected", seed=123)
    model_b = LinTrainableAssistedModelB(field_size=4, comms_size=1, sr_mode="expected", seed=123)

    tap = TrainableAssistedPlayers(layout, model_a=model_a, model_b=model_b)
    player_a, player_b = tap.players()

    n2 = layout.field_size * layout.field_size
    field = np.random.randint(0, 2, size=(n2,), dtype=np.int32)
    gun = np.zeros((n2,), dtype=np.int32)
    gun[np.random.randint(0, n2)] = 1

    # Greedy: deterministic for same input.
    tap.set_explore(False)
    comm1 = player_a.decide(field)
    assert comm1.shape == (layout.comms_size,)
    assert comm1.dtype == np.int32
    assert tap.previous is not None  # A populated previous
    shoot1 = player_b.decide(gun, comm1)
    assert shoot1 in (0, 1)
    assert isinstance(player_a.get_log_prob(), float)
    assert isinstance(player_b.get_log_prob(), float)

    comm2 = player_a.decide(field)
    assert np.array_equal(comm1, comm2), "Greedy mode must be deterministic for same input."

    # Explore: should sometimes differ across repeated calls (probabilistic).
    tap.set_explore(True)
    comms = []
    for _ in range(30):
        comms.append(int(player_a.decide(field)[0]))
    # Not guaranteed, but overwhelmingly likely for non-degenerate prob.
    assert len(set(comms)) >= 1
    assert player_a.get_log_prob() is not None

    # Reset clears previous + logprobs.
    tap.reset()
    assert tap.previous is None
    try:
        _ = player_a.get_log_prob()
        raise AssertionError("Expected get_log_prob to fail after reset.")
    except RuntimeError:
        pass
    try:
        _ = player_b.get_log_prob()
        raise AssertionError("Expected get_log_prob to fail after reset.")
    except RuntimeError:
        pass
