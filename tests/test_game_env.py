import sys
import numpy as np
import pytest
sys.path.append("./src")

@pytest.mark.usefixtures("qsb")
def test_game_env_reset_and_provide_shapes_and_copies():
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout(field_size=4, comms_size=1, enemy_probability=0.5, channel_noise=0.0)
    env = GameEnv(layout)

    np.random.seed(0)
    env.reset()
    assert env.field.shape == (layout.field_size, layout.field_size)
    assert env.gun.shape == (layout.field_size, layout.field_size)
    assert set(np.unique(env.field)).issubset({0, 1})
    assert int(env.gun.sum()) == 1

    field1, gun1 = env.provide()
    assert field1.shape == (layout.field_size ** 2,)
    assert gun1.shape == (layout.field_size ** 2,)

    # Returned values must be copies (mutating them must not mutate internal state).
    field1[0] = 1 - field1[0]
    gun1[0] = 1 - gun1[0]
    field2, gun2 = env.provide()
    assert field2[0] != field1[0] or gun2[0] != gun1[0]


@pytest.mark.usefixtures("qsb")
def test_game_env_provide_requires_reset():
    from Q_Sea_Battle.game_env import GameEnv

    env = GameEnv()
    with pytest.raises(RuntimeError):
        env.provide()


@pytest.mark.usefixtures("qsb")
def test_game_env_evaluate_requires_reset_and_one_hot_gun():
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout(field_size=4, comms_size=1)
    env = GameEnv(layout)

    with pytest.raises(RuntimeError):
        env.evaluate(0)

    np.random.seed(0)
    env.reset()

    # Force a broken gun and ensure evaluate complains.
    env.gun[:] = 0
    with pytest.raises(RuntimeError):
        env.evaluate(0)


@pytest.mark.usefixtures("qsb")
def test_game_env_evaluate_reward_matches_cell_value():
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout(field_size=4, comms_size=1)
    env = GameEnv(layout)

    np.random.seed(1)
    env.reset()

    field, gun = env.provide()
    gun_idx = int(np.argmax(gun))
    cell_value = int(field[gun_idx])

    assert env.evaluate(cell_value) == 1.0
    assert env.evaluate(1 - cell_value) == 0.0


@pytest.mark.usefixtures("qsb")
def test_game_env_apply_channel_noise_edge_cases():
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.game_layout import GameLayout

    comm = np.array([0, 1, 1, 0], dtype=int)

    env0 = GameEnv(GameLayout(field_size=4, comms_size=4, channel_noise=0.0))
    out0 = env0.apply_channel_noise(comm)
    assert np.array_equal(out0, comm)
    assert out0 is not comm  # should be a copy

    env1 = GameEnv(GameLayout(field_size=4, comms_size=4, channel_noise=1.0))
    out1 = env1.apply_channel_noise(comm)
    assert np.array_equal(out1, 1 - comm)


@pytest.mark.usefixtures("qsb")
def test_game_env_apply_channel_noise_probabilistic_is_seedable():
    from Q_Sea_Battle.game_env import GameEnv
    from Q_Sea_Battle.game_layout import GameLayout

    comm = np.array([0, 1, 1, 0], dtype=int)
    env = GameEnv(GameLayout(field_size=4, comms_size=len(comm), channel_noise=0.5))

    np.random.seed(0)
    out_a = env.apply_channel_noise(comm)
    np.random.seed(0)
    out_b = env.apply_channel_noise(comm)
    assert np.array_equal(out_a, out_b)
