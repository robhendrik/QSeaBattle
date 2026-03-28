import dataclasses
import pytest
import sys
sys.path.append("./src")

@pytest.mark.usefixtures("qsb")
def test_game_layout_defaults():
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout()
    assert isinstance(layout.field_size, int)
    assert layout.field_size > 0
    assert isinstance(layout.comms_size, int)
    assert layout.comms_size > 0
    assert 0.0 <= float(layout.enemy_probability) <= 1.0
    assert 0.0 <= float(layout.channel_noise) <= 1.0
    assert isinstance(layout.number_of_games_in_tournament, int)
    assert layout.number_of_games_in_tournament > 0

    # Log columns should at least include the core outputs.
    assert isinstance(layout.log_columns, list)
    for col in ["field", "gun", "comm", "shoot", "reward"]:
        assert col in layout.log_columns


@pytest.mark.usefixtures("qsb")
def test_game_layout_is_immutable():
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        layout.field_size = 8


@pytest.mark.usefixtures("qsb")
@pytest.mark.parametrize(
    "kwargs",
    [
        {"field_size": 0},
        {"field_size": -1},
        {"field_size": 3},  # 3^2 = 9 not power of two
        {"comms_size": 0},
        {"comms_size": -1},
        {"field_size": 4, "comms_size": 3},  # 16 % 3 != 0
        {"enemy_probability": -0.01},
        {"enemy_probability": 1.01},
        {"channel_noise": -0.01},
        {"channel_noise": 1.01},
        {"number_of_games_in_tournament": 0},
    ],
)
def test_game_layout_invalid_values_raise(kwargs):
    from Q_Sea_Battle.game_layout import GameLayout

    with pytest.raises((TypeError, ValueError)):
        GameLayout(**kwargs)


@pytest.mark.usefixtures("qsb")
def test_game_layout_from_dict_ignores_unknown_keys():
    from Q_Sea_Battle.game_layout import GameLayout

    d = {
        "field_size": 4,
        "comms_size": 2,
        "enemy_probability": 0.25,
        "channel_noise": 0.1,
        "number_of_games_in_tournament": 7,
        "unknown_key": "ignored",
    }
    layout = GameLayout.from_dict(d)
    assert layout.field_size == 4
    assert layout.comms_size == 2
    assert float(layout.enemy_probability) == 0.25
    assert float(layout.channel_noise) == 0.1
    assert layout.number_of_games_in_tournament == 7


@pytest.mark.usefixtures("qsb")
def test_game_layout_to_dict_roundtrip_core_fields():
    from Q_Sea_Battle.game_layout import GameLayout

    layout = GameLayout(field_size=8, comms_size=4, enemy_probability=0.4, channel_noise=0.2)
    d = layout.to_dict()
    for key in [
        "field_size",
        "comms_size",
        "enemy_probability",
        "channel_noise",
        "number_of_games_in_tournament",
        "log_columns",
    ]:
        assert key in d
    assert d["field_size"] == 8
    assert d["comms_size"] == 4
