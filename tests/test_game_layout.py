"""Tests for the GameLayout configuration class."""

from __future__ import annotations
import pytest
import sys  
sys.path.append("./src")
from Q_Sea_Battle.game_layout import GameLayout


def test_default_layout_is_valid() -> None:
    """Default constructor should produce a valid layout."""
    layout = GameLayout()

    assert layout.field_size == 4
    assert layout.comms_size == 1
    assert layout.enemy_probability == 0.5
    assert layout.channel_noise == 0.0
    assert layout.number_of_games_in_tournament == 100

    # n2 should be power of two.
    n2 = layout.field_size ** 2
    assert n2 > 0 and (n2 & (n2 - 1)) == 0

    # Log columns should contain at least the core fields.
    for col in ("field", "gun", "comm", "shoot", "reward"):
        assert col in layout.log_columns


def test_invalid_field_size_not_power_of_two() -> None:
    """field_size that does not produce n2 being a power of 2 should fail."""
    with pytest.raises(ValueError):
        GameLayout(field_size=3)  # 3**2 = 9, not a power of two.


def test_invalid_comms_not_dividing_n2() -> None:
    """comms_size must divide n2."""
    with pytest.raises(ValueError):
        GameLayout(field_size=4, comms_size=3)  # 16 % 3 != 0


def test_invalid_enemy_probability() -> None:
    """enemy_probability must be in [0, 1]."""
    with pytest.raises(ValueError):
        GameLayout(enemy_probability=1.1)
    with pytest.raises(ValueError):
        GameLayout(enemy_probability=-0.1)


def test_invalid_channel_noise() -> None:
    """channel_noise must be in [0, 1]."""
    with pytest.raises(ValueError):
        GameLayout(channel_noise=1.5)
    with pytest.raises(ValueError):
        GameLayout(channel_noise=-0.2)


def test_invalid_number_of_games() -> None:
    """number_of_games_in_tournament must be positive."""
    with pytest.raises(ValueError):
        GameLayout(number_of_games_in_tournament=0)
    with pytest.raises(ValueError):
        GameLayout(number_of_games_in_tournament=-5)


def test_from_dict_overrides_and_ignores_unknown_keys() -> None:
    """from_dict should override known fields and ignore unknown ones."""
    params = {
        "field_size": 8,
        "enemy_probability": 0.7,
        "unknown_key": "ignored",
    }

    layout = GameLayout.from_dict(params)

    assert layout.field_size == 8
    assert layout.enemy_probability == 0.7
    # Default values should be used for the rest.
    assert layout.comms_size == GameLayout().comms_size


def test_to_dict_and_from_dict_roundtrip() -> None:
    """to_dict output should be usable with from_dict to reconstruct layout."""
    original = GameLayout(
        field_size=8,
        comms_size=2,
        enemy_probability=0.3,
        channel_noise=0.1,
        number_of_games_in_tournament=50,
    )

    as_dict = original.to_dict()
    reconstructed = GameLayout.from_dict(as_dict)

    assert reconstructed == original
