"""Tournament orchestration for QSeaBattle.

Author: Rob Hendriks
Version: 0.1
"""

from __future__ import annotations

from .game import Game
from .game_env import GameEnv
from .game_layout import GameLayout
from .players_base import Players
from .tournament_log import TournamentLog


class Tournament:
    """Run a multi-game QSeaBattle tournament.

    This class repeatedly runs Game.play() and records the outcomes
    in a TournamentLog instance.
    """

    def __init__(
        self, game_env: GameEnv, players: Players, game_layout: GameLayout
    ) -> None:
        """Initialise a tournament runner.

        Args:
            game_env: Game environment instance.
            players: Players factory for A and B.
            game_layout: Configuration specifying tournament length.
        """
        self.game_env = game_env
        self.players = players
        self.game_layout = game_layout

    def tournament(self) -> TournamentLog:
        """Execute a full tournament and return its log.

        The same GameEnv and Players instances are reused across games,
        with reset between games. Any optional log-probabilities or
        previous-measurement data provided by specialised players are
        recorded when available.

        Returns:
            A TournamentLog instance containing all game results.
        """
        log = TournamentLog(self.game_layout)
        game = Game(self.game_env, self.players)

        n_games = self.game_layout.number_of_games_in_tournament
        # For now we use fixed tournament_id and meta_id; these can be
        # extended later if needed.
        tournament_id = 0
        meta_id = 0

        for game_id in range(n_games):
            # Run a single game.
            reward, field, gun, comm, shoot = game.play()
            cell_value = int(field[gun == 1][0])

            # Basic outcome logging.
            log.update(field, gun, comm, shoot, cell_value, reward)

            # Optional: log-probabilities if provided by players.
            if getattr(self.players, "has_log_probs", False):
                player_a, player_b = self.players.players()
                # Assume child players implement get_log_prob().
                logprob_comm = player_a.get_log_prob()
                logprob_shoot = player_b.get_log_prob()
                log.update_log_probs(logprob_comm, logprob_shoot)

            # Optional: previous measurements/outcomes if provided.
            if getattr(self.players, "has_prev", False):
                player_a, _ = self.players.players()
                prev = player_a.get_prev()
                if prev is not None:
                    prev_meas, prev_out = prev
                    log.update_log_prev(prev_meas, prev_out)

            # Add identifiers for this game.
            log.update_indicators(game_id=game_id, tournament_id=tournament_id, meta_id=meta_id)

        return log

