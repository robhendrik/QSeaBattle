"""Game layout configuration for QSeaBattle.

This module defines :class:`GameLayout`, an immutable dataclass that captures the
parameters for a single QSeaBattle game and its surrounding tournament setup.

Notes:
    * The playing field is a square grid with side length ``field_size``.
      Internally, the field is often treated as a flattened vector with length
      ``n2 = field_size ** 2``.
    * ``n2`` is required to be a power of two. This constraint is typically used
      by downstream components that rely on bit/partition-friendly dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class GameLayout:
    """Immutable configuration for a QSeaBattle game.

    The layout specifies board dimensions, communication size, probabilities used
    by the game generator and channel model, and the tournament log schema.
    Instances are validated on creation and are intended to be treated as
    read-only.

    Attributes:
        field_size: Side length ``n`` of the square field. The flattened field
            has length ``n2 = field_size ** 2`` and ``n2`` must be a power of two.
        comms_size: Length ``m`` of the communication vector. Must be a positive
            divisor of ``n2``.
        enemy_probability: Probability that a generated field cell equals 1.
            Must be in ``[0.0, 1.0]``.
        channel_noise: Bit-flip probability for the channel. Must be in
            ``[0.0, 1.0]``.
        number_of_games_in_tournament: Number of games played per tournament.
            Must be a positive integer.
        log_columns: Column names used when logging tournament/game events.
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
        """Validate parameters after dataclass initialization.

        Raises:
            TypeError: If a field has an unexpected type.
            ValueError: If a field violates a basic constraint (e.g., invalid
                probability range, incompatible sizes).
        """
        # Basic type checks for the core integer parameters. These are validated
        # explicitly because other components assume integer arithmetic (e.g.,
        # modulus checks and bitwise operations).
        if not isinstance(self.field_size, int):
            raise TypeError("field_size must be an int.")
        if not isinstance(self.comms_size, int):
            raise TypeError("comms_size must be an int.")
        if not isinstance(self.number_of_games_in_tournament, int):
            raise TypeError("number_of_games_in_tournament must be an int.")

        if self.field_size <= 0:
            raise ValueError("field_size must be a positive integer.")

        n2 = self.field_size ** 2

        # Require n2 to be a power of two: n2 = 2^k. This is checked using the
        # standard bit trick for positive integers.
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
        """Create a validated :class:`GameLayout` from a mapping.

        Unknown keys are ignored. Missing keys fall back to dataclass defaults.
        The returned instance is validated via :meth:`__post_init__`.

        Args:
            parameters: Mapping of field names to override values.

        Returns:
            A new validated layout instance.
        """
        allowed_keys = set(cls.__dataclass_fields__.keys())
        filtered: Dict = {
            key: value for key, value in parameters.items() if key in allowed_keys
        }
        return cls(**filtered)

    def to_dict(self) -> Dict:
        """Convert this layout to a dictionary.

        Returns:
            A dictionary containing all dataclass fields and their current values.
        """
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        """Check whether an integer is a positive power of two.

        Args:
            value: Integer to test.

        Returns:
            ``True`` if ``value`` is a positive power of two, otherwise ``False``.
        """
        return value > 0 and (value & (value - 1)) == 0