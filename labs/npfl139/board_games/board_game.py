# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import enum
import numpy as np


class BoardGame:
    class Outcome(enum.Enum):
        LOSS = 1
        DRAW = 2
        WIN = 3

        def reverse(self):
            """Reverse the outcome."""
            return BoardGame.Outcome(4 - self.value)

    ACTIONS: int  # Number of possible actions
    N: int  # Size of the board
    C: int  # Number of channels in the board representation

    def clone(self, swap_players=False) -> "BoardGame":
        """Clone the game state, optionally swapping the players."""

    @property
    def board(self) -> np.ndarray:
        """Return the internal representation of board as a NxN numpy array."""

    @property
    def board_features(self) -> np.ndarray:
        """Return the board as a NxNxC numpy array of features."""

    @property
    def to_play(self) -> int:
        """Return the current player."""

    def outcome(self, player: int) -> Outcome | None:
        """Return the game outcome for a given player."""

    def valid(self, action: int) -> bool:
        """Return whether the given action is valid."""

    def valid_actions(self) -> list[int]:
        """Return the list of valid actions."""

    def move(self, action: int) -> None:
        """Execute the given action."""

    # Static factory methods
    @staticmethod
    def from_name(game_name: str) -> type["BoardGame"]:
        """Return a game class from the game name."""
        assert game_name in BoardGame._games, f"BoardGame.from_name got unknown game name: {game_name}"
        return BoardGame._games[game_name]

    @staticmethod
    def register_game(game_name: str, game_class: type["BoardGame"]) -> None:
        """Register a new game class."""
        BoardGame._games[game_name] = game_class

    _games: dict[str, type["BoardGame"]] = {}

    # A game might provide reference players.
    @staticmethod
    def player_from_name(self, player_name: str):
        """Return a player object for the given player name."""
