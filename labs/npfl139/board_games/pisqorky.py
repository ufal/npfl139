# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np

from .board_game import BoardGame
from .board_game_player import BoardGamePlayer


class Pisqorky(BoardGame):
    ACTIONS = 225
    """Number of actions in the game."""
    N = 15
    """The number of rows and columns in the game board."""
    C = 3
    """The number of features in the board representation."""

    def __init__(self):
        self._board = np.zeros([self.N, self.N], dtype=np.int8)
        self._to_play = 0
        self._outcome = None
        self._screen = None
        self._last_action, self._winning_stones = None, []

    def clone(self, swap_players=False) -> "Pisqorky":
        clone = Pisqorky()
        if swap_players:
            clone._board[:, :] = (self._board == 2) + 2 * (self._board == 1)
            clone._to_play = 1 - self._to_play
            clone._outcome = self._outcome.reverse() if self._outcome is not None else None
        else:
            clone._board[:, :] = self._board
            clone._to_play = self._to_play
            clone._outcome = self._outcome
        clone._last_action, clone._winning_stones = self._last_action, self._winning_stones
        return clone

    @property
    def board(self) -> np.ndarray:
        return self._board

    @property
    def board_features(self) -> np.ndarray:
        return np.eye(3, dtype=np.uint8)[self._board]

    @property
    def to_play(self) -> int:
        return self._to_play

    def outcome(self, player: int) -> BoardGame.Outcome | None:
        return self._outcome if self._outcome is None or player == self._to_play else self._outcome.reverse()

    def valid(self, action: int) -> bool:
        return self._outcome is None and action >= 0 and action < self.ACTIONS \
            and self._board[action // self.N, action % self.N] == 0

    def valid_actions(self) -> list[int]:
        return np.nonzero(self._board.ravel() == 0)[0] if self._outcome is None else []

    def move(self, action: int) -> bool:
        if not self.valid(action):
            raise ValueError("An invalid action to Pisqorky.move")
        self._last_action = (action // self.N, action % self.N)
        self._board[self._last_action] = self._to_play + 1
        self._to_play = 1 - self._to_play

        # Check for a winner
        free_fields = False
        for y in range(self.N):
            for x in range(self.N):
                if self._board[y, x] == 0:
                    free_fields = True
                    continue
                field = self._board[y, x]
                if (x >= 4 and y + 4 < self.N and field ==
                        self._board[y + 1, x - 1] == self._board[y + 2, x - 2] ==
                        self._board[y + 3, x - 3] == self._board[y + 4, x - 4]):
                    self._outcome = self.Outcome.WIN if field - 1 == self._to_play else self.Outcome.LOSS
                    self._winning_stones = [(y + i, x - i) for i in range(5)]
                    return
                if (y + 4 < self.N and field ==
                        self._board[y + 1, x] == self._board[y + 2, x] ==
                        self._board[y + 3, x] == self._board[y + 4, x]):
                    self._outcome = self.Outcome.WIN if field - 1 == self._to_play else self.Outcome.LOSS
                    self._winning_stones = [(y + i, x) for i in range(5)]
                    return
                if (x + 4 < self.N and y + 4 < self.N and field ==
                        self._board[y + 1, x + 1] == self._board[y + 2, x + 2] ==
                        self._board[y + 3, x + 3] == self._board[y + 4, x + 4]):
                    self._outcome = self.Outcome.WIN if field - 1 == self._to_play else self.Outcome.LOSS
                    self._winning_stones = [(y + i, x + i) for i in range(5)]
                    return
                if (x + 4 < self.N and field ==
                        self._board[y, x + 1] == self._board[y, x + 2] ==
                        self._board[y, x + 3] == self._board[y, x + 4]):
                    self._outcome = self.Outcome.WIN if field - 1 == self._to_play else self.Outcome.LOSS
                    self._winning_stones = [(y, x + i) for i in range(5)]
                    return
        if not free_fields:
            self._outcome = self.Outcome.DRAW

    def render(self):
        import pygame
        A, W = 40, self.N * 40
        white, black, red, blue, lred, lblue = \
            (255, 255, 255), (0, 0, 0), (224, 32, 32), (32, 32, 224), (224, 128, 128), (128, 128, 224)

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.get_surface() or pygame.display.set_mode((W, W))
            self._screen_surface = pygame.Surface((W, W))

        self._screen_surface.fill(white)
        for y in range(self.N + 1):
            pygame.draw.line(self._screen_surface, black, (0, max(0, min(W - 2, y * A - 1))), (W, max(0, min(W - 2, y * A - 1))), 2)
        for x in range(self.N + 1):
            pygame.draw.line(self._screen_surface, black, (max(0, min(W - 2, x * A - 1)), 0), (max(0, min(W - 2, x * A - 1)), W), 2)
        for y in range(self.N):
            for x in range(self.N):
                if (y, x) == self._last_action or (y, x) in self._winning_stones:
                    pygame.draw.rect(self._screen_surface, lblue if self._board[y, x] == 1 else lred,
                                     (x * A + 1 + (x == 0), y * A + 1 + (y == 0),
                                      A - 2 - (x == 0) - (x + 1 == self.N), A - 2 - (y == 0) - (y + 1 == self.N)))
                if self._board[y, x] == 1:
                    pygame.draw.circle(self._screen_surface, blue, (x * A + A // 2, y * A + A // 2), A // 2 - 3, 5)
                if self._board[y, x] == 2:
                    pygame.draw.line(self._screen_surface, red, (x * A + 6, y * A + 6), (x * A + A - 7, y * A + A - 7), 7)
                    pygame.draw.line(self._screen_surface, red, (x * A + 6, y * A + A - 7), (x * A + A - 7, y * A + 6), 7)

        self._screen.blit(self._screen_surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()

    def mouse_input(self):
        import pygame
        A = 40

        self.render()
        if self._outcome is not None:
            raise ValueError("The Pisqorky game is over in Pisqorky.mouse_input")

        chosen = None
        while chosen is None:
            event = pygame.event.wait()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                chosen = (y // A) * self.N + x // A
                if not self.valid(chosen):
                    chosen = None
            if event.type == pygame.WINDOWEXPOSED:
                self._screen.blit(self._screen_surface, (0, 0))
                pygame.display.flip()
            if event.type == pygame.QUIT:
                print("Window closed, stopping application.")
                exit()

        return chosen

    @staticmethod
    def player_from_name(player_name: str) -> type[BoardGamePlayer["Pisqorky"]]:
        assert player_name in Pisqorky._players, f"Pisqorky.player_from_name got unknown player name: {player_name}"
        return Pisqorky._players[player_name]

    @staticmethod
    def register_player(player_name: str, player: type[BoardGamePlayer["Pisqorky"]]):
        Pisqorky._players[player_name] = player

    _players: dict[str, type[BoardGamePlayer["Pisqorky"]]] = {}

# Register the Pisqorky game.
BoardGame.register_game("pisqorky", Pisqorky)
