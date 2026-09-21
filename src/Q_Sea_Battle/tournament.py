"""Tournament orchestration for QSeaBattle.

This module provides a thin orchestration layer that repeatedly runs games and
records per-game outcomes into a :class:`~.tournament_log.TournamentLog`.

The tournament runner optionally records additional per-game metadata when the
configured players expose it (e.g., action log-probabilities or previous
measurement/outcome data).
"""

from __future__ import annotations

from .game import Game
from .game_env import GameEnv
from .game_layout import GameLayout
from .players_base import Players
from .tournament_log import TournamentLog


class Tournament:
    """Run a multi-game QSeaBattle tournament.

    A tournament consists of repeatedly calling :meth:`~.game.Game.play` for
    ``game_layout.number_of_games_in_tournament`` games and appending the results
    to a :class:`~.tournament_log.TournamentLog`.

    The same :class:`~.game_env.GameEnv` and :class:`~.players_base.Players`
    instances are reused across games; per-game reset behavior is handled by the
    underlying objects (e.g., inside :meth:`~.game.Game.play`).
    """

    def __init__(
        self, game_env: GameEnv, players: Players, game_layout: GameLayout
    ) -> None:
        """Initialize the tournament runner.

        Args:
            game_env: Game environment used to execute games.
            players: Factory/container providing player A and player B instances.
            game_layout: Layout/configuration specifying the number of games and
                other tournament-level settings.
        """
        self.game_env = game_env
        self.players = players
        self.game_layout = game_layout

    def tournament(self) -> TournamentLog:
        """Execute the tournament and return the accumulated log.

        For each game, the runner logs the returned game artifacts and computes
        the selected cell value from the returned ``field`` and ``gun`` tensors.

        Optional player-provided metadata is recorded when available:
        - If ``players.has_log_probs`` is truthy, the runner calls
          ``player_a.get_log_prob()`` and ``player_b.get_log_prob()`` and stores
          the returned values.
        - If ``players.has_prev`` is truthy, the runner calls
          ``player_a.get_prev()`` and, if non-``None``, stores the returned
          ``(prev_meas, prev_out)`` pair.

        Returns:
            TournamentLog: Log containing results for all games in the tournament.
        """
        log = TournamentLog(self.game_layout)
        game = Game(self.game_env, self.players)

        n_games = self.game_layout.number_of_games_in_tournament
        # Currently fixed identifiers; kept as placeholders for potential
        # extensions such as multiple tournaments or meta-experiments.
        tournament_id = 0
        meta_id = 0

        for game_id in range(n_games):
            reward, field, gun, comm, shoot = game.play()

            # Determine the "hit" cell value by selecting the field entry at the
            # gun location. Assumes exactly one gun cell is marked with 1.
            cell_value = int(field[gun == 1][0])

            # Record the main game outputs.
            log.update(field, gun, comm, shoot, cell_value, reward)

            # Record optional log-probabilities when supported by the players.
            if getattr(self.players, "has_log_probs", False):
                player_a, player_b = self.players.players()
                # Contract: child players may implement get_log_prob().
                logprob_comm = player_a.get_log_prob()
                logprob_shoot = player_b.get_log_prob()
                log.update_log_probs(logprob_comm, logprob_shoot)

            # Record optional "previous measurement/outcome" metadata when present.
            if getattr(self.players, "has_prev", False):
                player_a, _ = self.players.players()
                prev = player_a.get_prev()
                if prev is not None:
                    prev_meas, prev_out = prev
                    log.update_log_prev(prev_meas, prev_out)

            # Add identifiers for this game.
            log.update_indicators(
                game_id=game_id, tournament_id=tournament_id, meta_id=meta_id
            )

        return log