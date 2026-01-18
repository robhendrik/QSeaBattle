"""Game layout dataclass and configuration for QSeaBattle.

Author: Rob Hendriks
Package: Q_Sea_Battle
Version: 0.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class GameLayout:
    """Immutable configuration for a QSeaBattle game.

    This dataclass holds the parameters that define a single game and
    its surrounding tournament configuration. Instances are validated
    at creation time and treated as read-only.

    Attributes:
        field_size: Size of one dimension of the square field ``n``.
            The flattened field has length ``n2 = field_size ** 2``. The
            value of ``n2`` must be a power of 2.
        comms_size: Length of the communication vector ``m``. Must
            divide ``n2``.
        enemy_probability: Probability that a cell in the field equals 1.
            Must lie in the interval [0.0, 1.0].
        channel_noise: Probability that a bit is flipped in the channel.
            Must lie in the interval [0.0, 1.0].
        number_of_games_in_tournament: Number of games per tournament.
            Must be a positive integer.
        log_columns: List of column names for the tournament log.
    """

    field_size: int = 4
    comms_size: int = 1
    enemy_probability: float = 0.5
    channel_noise: float = 0.0
    number_of_games_in_tournament: int = 100
    log_columns: List[str] = field(
        default_factory=lambda: [
            "field",
            "gun",
            "comm",
            "shoot",
            "cell_value",
            "reward",
            "sample_weight",
            "logprob_comm",
            "logprob_shoot",
            "game_id",
            "tournament_id",
            "meta_id",
            "game_uid",
            "prev_measurements",
            "prev_outcomes",
        ]
    )

    def __post_init__(self) -> None:
        """Validate parameters after initialisation.

        Raises:
            TypeError: If types of core attributes are incorrect.
            ValueError: If values violate basic constraints from the spec.
        """
        # Basic type checks for the core integer parameters.
        if not isinstance(self.field_size, int):
            raise TypeError("field_size must be an int.")
        if not isinstance(self.comms_size, int):
            raise TypeError("comms_size must be an int.")
        if not isinstance(self.number_of_games_in_tournament, int):
            raise TypeError("number_of_games_in_tournament must be an int.")

        if self.field_size <= 0:
            raise ValueError("field_size must be a positive integer.")

        n2 = self.field_size ** 2

        # n2 must be a power of two (n2 = 2^k).
        if not self._is_power_of_two(n2):
            raise ValueError(
                f"field_size ** 2 must be a power of 2, got field_size={self.field_size}, n2={n2}."
            )

        # comms_size must be a positive divisor of n2.
        if self.comms_size <= 0:
            raise ValueError("comms_size must be a positive integer.")
        if n2 % self.comms_size != 0:
            raise ValueError(
                f"comms_size must divide field_size ** 2; got comms_size={self.comms_size}, n2={n2}."
            )

        # Probabilities must lie in [0, 1].
        if not (0.0 <= self.enemy_probability <= 1.0):
            raise ValueError(
                f"enemy_probability must be in [0.0, 1.0], got {self.enemy_probability}."
            )
        if not (0.0 <= self.channel_noise <= 1.0):
            raise ValueError(
                f"channel_noise must be in [0.0, 1.0], got {self.channel_noise}."
            )

        if self.number_of_games_in_tournament <= 0:
            raise ValueError("number_of_games_in_tournament must be > 0.")

        # Minimal check that log_columns is a list of strings.
        if not isinstance(self.log_columns, list) or not all(
            isinstance(col, str) for col in self.log_columns
        ):
            raise TypeError("log_columns must be a list of strings.")

    @classmethod
    def from_dict(cls, parameters: Dict) -> "GameLayout":
        """Create a GameLayout instance from a dictionary of parameters.

        Unknown keys in the input dictionary are ignored. Missing keys
        are filled with the dataclass defaults, and the resulting
        instance is validated via ``__post_init__``.

        Args:
            parameters: Dictionary with parameter overrides.

        Returns:
            A new validated GameLayout instance.
        """
        allowed_keys = set(cls.__dataclass_fields__.keys())
        filtered: Dict = {
            key: value for key, value in parameters.items() if key in allowed_keys
        }
        return cls(**filtered)

    def to_dict(self) -> Dict:
        """Return a dictionary representation of this layout.

        The dictionary contains all dataclass fields and their current
        values.

        Returns:
            A dictionary with all layout parameters.
        """
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        """Return True if value is a power of two.

        Args:
            value: Integer to test.

        Returns:
            True if ``value`` is a positive power of two, False otherwise.
        """
        return value > 0 and (value & (value - 1)) == 0

