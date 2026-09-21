"""Tournament logging utilities for QSeaBattle.

This module provides :class:`TournamentLog`, a thin wrapper around a Pandas
``DataFrame`` used to store one row per played game in a tournament.

The set of columns is defined by :class:`~.game_layout.GameLayout` via
``game_layout.log_columns``. This class appends rows and then updates selected
fields (e.g., log-probabilities and identifiers) for the most recently added row.
"""

from __future__ import annotations

import uuid
from typing import Any, Tuple

import numpy as np
import pandas as pd

from .game_layout import GameLayout


class TournamentLog:
    """Structured log for storing QSeaBattle tournament results.

    The log is stored as a Pandas ``DataFrame`` with one row per game. A typical
    usage pattern is:

    1) Call :meth:`update` to append a new row with core game outputs.
    2) Call one or more of the ``update_*`` methods to fill in additional fields
       for that same last row (e.g., identifiers, log-probabilities).

    Attributes:
        game_layout: Layout instance defining the log column names.
        log: Pandas DataFrame containing one row per logged game.
    """

    def __init__(self, game_layout: GameLayout) -> None:
        """Initialise an empty tournament log.

        Args:
            game_layout: Layout providing the log column names.
        """
        self.game_layout = game_layout
        self.log = pd.DataFrame(columns=game_layout.log_columns)

    # --------------------------------------------------------------------- #
    # Row update helpers
    # --------------------------------------------------------------------- #

    def update(
        self,
        field: np.ndarray,
        gun: np.ndarray,
        comm: np.ndarray,
        shoot: int,
        cell_value: int,
        reward: float,
    ) -> None:
        """Append a new game result row to the log.

        This method appends a row containing the primary per-game artifacts and
        sets optional/late-bound fields to ``None``. Those fields can be filled
        in later by calling the corresponding ``update_*`` methods.

        Args:
            field: Game field state for the game.
            gun: Gun state/action representation for the game.
            comm: Communication representation for the game.
            shoot: Shot/cell index selected for the game.
            cell_value: Observed value at the shot cell.
            reward: Scalar reward for the game.
        """
        row = {
            "field": field,
            "gun": gun,
            "comm": comm,
            "shoot": int(shoot),
            "cell_value": int(cell_value),
            "reward": float(reward),
            "logprob_comm": None,
            "logprob_shoot": None,
            "game_id": None,
            "tournament_id": None,
            "meta_id": None,
            "game_uid": None,
            "prev_measurements": None,
            "prev_outcomes": None,
        }

        # Assigning via .loc avoids deprecated/inefficient DataFrame.append.
        self.log.loc[len(self.log)] = row

    def _last_row_index(self) -> int:
        """Return the index of the last logged row.

        Returns:
            Integer index label of the last row.

        Raises:
            RuntimeError: If no rows have been logged yet.
        """
        if self.log.empty:
            raise RuntimeError("TournamentLog is empty; no rows to update.")
        return int(self.log.index[-1])

    def update_log_probs(self, logprob_comm: float, logprob_shoot: float) -> None:
        """Update log-probabilities for the last logged game.

        Args:
            logprob_comm: Log-probability associated with the communication
                decision.
            logprob_shoot: Log-probability associated with the shooting
                decision.
        """
        idx = self._last_row_index()
        self.log.at[idx, "logprob_comm"] = float(logprob_comm)
        self.log.at[idx, "logprob_shoot"] = float(logprob_shoot)

    def update_log_prev(self, prev_meas: Any, prev_out: Any) -> None:
        """Update previous measurements/outcomes for the last logged game.

        These fields store per-layer history and are treated as opaque objects by
        the logger.

        Args:
            prev_meas: Previous measurements per shared layer.
            prev_out: Previous outcomes per shared layer.
        """
        idx = self._last_row_index()
        self.log.at[idx, "prev_measurements"] = prev_meas
        self.log.at[idx, "prev_outcomes"] = prev_out

    def update_indicators(self, game_id: int, tournament_id: int, meta_id: int) -> None:
        """Update identifier fields for the last logged game.

        In addition to setting ``game_id``, ``tournament_id``, and ``meta_id``,
        this method also generates a unique ``game_uid`` string.

        Args:
            game_id: Identifier of the game within a tournament.
            tournament_id: Identifier of the tournament.
            meta_id: Identifier for experimental metadata.
        """
        idx = self._last_row_index()
        self.log.at[idx, "game_id"] = int(game_id)
        self.log.at[idx, "tournament_id"] = int(tournament_id)
        self.log.at[idx, "meta_id"] = int(meta_id)
        # UUID4 hex string provides a unique identifier per logged game.
        self.log.at[idx, "game_uid"] = uuid.uuid4().hex

    # --------------------------------------------------------------------- #
    # Summary statistics
    # --------------------------------------------------------------------- #

    def outcome(self) -> Tuple[float, float]:
        """Compute aggregate reward statistics over the logged games.

        The returned values are computed from the ``reward`` column:
        - Mean reward: arithmetic mean across games.
        - Standard error: sample standard deviation (ddof=1) divided by
          ``sqrt(n)``.

        Returns:
            Tuple of (mean_reward, std_error). Returns (0.0, 0.0) if the log is
            empty.
        """
        if self.log.empty:
            return 0.0, 0.0

        rewards = self.log["reward"].astype(float).to_numpy()
        mean_reward = float(rewards.mean())

        n = rewards.size
        if n <= 1:
            std_error = 0.0
        else:
            std = float(rewards.std(ddof=1))
            std_error = std / float(np.sqrt(n))

        return mean_reward, std_error