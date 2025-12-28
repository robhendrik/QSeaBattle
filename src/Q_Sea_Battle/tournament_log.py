"""Tournament logging utilities for QSeaBattle.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

import uuid
from typing import Any, Tuple

import numpy as np
import pandas as pd

from .game_layout import GameLayout


class TournamentLog:
    """Structured log for storing QSeaBattle tournament results.

    Attributes:
        game_layout: Layout that defines log columns.
        log: Pandas DataFrame containing one row per game.
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
        """Append a new game result to the log."""
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

        # Safe and warning-free append
        self.log.loc[len(self.log)] = row


    def _last_row_index(self) -> int:
        """Return the index of the last logged row.

        Returns:
            Integer index of the last row.

        Raises:
            RuntimeError: If no rows have been logged yet.
        """
        if self.log.empty:
            raise RuntimeError("TournamentLog is empty; no rows to update.")
        return int(self.log.index[-1])

    def update_log_probs(self, logprob_comm: float, logprob_shoot: float) -> None:
        """Update log-probabilities for the last logged game.

        Args:
            logprob_comm: Log-probability for the communication decision.
            logprob_shoot: Log-probability for the shoot decision.
        """
        idx = self._last_row_index()
        self.log.at[idx, "logprob_comm"] = float(logprob_comm)
        self.log.at[idx, "logprob_shoot"] = float(logprob_shoot)

    def update_log_prev(self, prev_meas: Any, prev_out: Any) -> None:
        """Update previous measurements/outcomes for the last game.

        Args:
            prev_meas: Previous measurements per shared layer.
            prev_out: Previous outcomes per shared layer.
        """
        idx = self._last_row_index()
        self.log.at[idx, "prev_measurements"] = prev_meas
        self.log.at[idx, "prev_outcomes"] = prev_out

    def update_indicators(
        self, game_id: int, tournament_id: int, meta_id: int
    ) -> None:
        """Update identifier fields for the last logged game.

        Also generates a unique game_uid string.

        Args:
            game_id: Identifier of the game within a tournament.
            tournament_id: Identifier of the tournament.
            meta_id: Identifier for experimental metadata.
        """
        idx = self._last_row_index()
        self.log.at[idx, "game_id"] = int(game_id)
        self.log.at[idx, "tournament_id"] = int(tournament_id)
        self.log.at[idx, "meta_id"] = int(meta_id)
        # Use UUID4 to generate a unique identifier per game.
        self.log.at[idx, "game_uid"] = uuid.uuid4().hex

    # --------------------------------------------------------------------- #
    # Summary statistics
    # --------------------------------------------------------------------- #

    def outcome(self) -> Tuple[float, float]:
        """Compute aggregate statistics over the tournament.

        The mean reward is the average of the "reward" column. The
        standard error is the sample standard deviation divided by the
        square root of the number of games.

        Returns:
            A tuple (mean_reward, std_error) summarising performance.
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

