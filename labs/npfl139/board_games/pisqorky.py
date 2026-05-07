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
        self._font = None
        self._last_action = None
        self._winning_stones = []

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
        clone._last_action = self._last_action
        clone._winning_stones = self._winning_stones
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

    def outcome(self, player: int | None = None) -> BoardGame.Outcome | None:
        if self._outcome is None or player is None or player == self._to_play:
            return self._outcome
        else:
            return self._outcome.reverse()

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

    def render_size(self):
        return self.N * self._A, self.N * self._A

    def render_to_surface(self, surface, weights=None, weight_max=None):
        import pygame

        white, black, red, blue, lred, lblue = \
            (255, 255, 255), (0, 0, 0), (224, 32, 32), (32, 32, 224), (224, 128, 128), (128, 128, 224)
        A, W = self._A, self.N * self._A

        if weights is not None:
            if self._font is None:
                pygame.freetype.init()
                self._font = pygame.freetype.SysFont(None, self._A // 2 - 2)
            weight_max = weight_max or max(weights[self.valid_actions()], default=1)

        surface.fill(black)
        surface.fill(white, (0, 0, *self.render_size()))
        for y in range(self.N + 1):
            pygame.draw.line(surface, black, (0, max(0, min(W - 2, y * A - 1))), (W, max(0, min(W - 2, y * A - 1))), 2)
        for x in range(self.N + 1):
            pygame.draw.line(surface, black, (max(0, min(W - 2, x * A - 1)), 0), (max(0, min(W - 2, x * A - 1)), W), 2)
        for y in range(self.N):
            for x in range(self.N):
                rect = pygame.Rect(x * A + 1 + (x == 0), y * A + 1 + (y == 0),
                                   A - 2 - (x == 0) - (x + 1 == self.N), A - 2 - (y == 0) - (y + 1 == self.N))
                if (y, x) == self._last_action or (y, x) in self._winning_stones:
                    pygame.draw.rect(surface, lblue if self._board[y, x] == 1 else lred, rect)
                if self._board[y, x] == 1:
                    pygame.draw.circle(surface, blue, (x * A + A // 2, y * A + A // 2), A // 2 - 3, 5)
                if self._board[y, x] == 2:
                    pygame.draw.line(surface, red, (x * A + 6, y * A + 6), (x * A + A - 7, y * A + A - 7), 7)
                    pygame.draw.line(surface, red, (x * A + 6, y * A + A - 7), (x * A + A - 7, y * A + 6), 7)
                if weights is not None and self.valid(y * self.N + x) and np.isfinite(weights[y * self.N + x]):
                    weight = weights[y * self.N + x] / weight_max if weight_max else 0.5
                    color = (255 * (1 - weight), 192 * weight, 0, 128)
                    pygame.gfxdraw.box(surface, rect, color)
                    font_rect = self._font.get_rect(f"{100 * weights[y * self.N + x]:04.1f}")
                    font_rect.center = rect.center
                    self._font.render_to(surface, font_rect, None, color[:3])

    def render(self):
        import pygame

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.get_surface() or pygame.display.set_mode(self.render_size())
            self._screen_surface = pygame.Surface(self.render_size())

        self.render_to_surface(self._screen_surface)

        self._screen.blit(self._screen_surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()

    def mouse_click_to_action(self, pos):
        x, y = pos
        chosen = (y // self._A) * self.N + x // self._A
        if self.valid(chosen):
            return chosen
        return None

    def mouse_input(self):
        import pygame

        self.render()
        if self._outcome is not None:
            raise ValueError("The Pisqorky game is over in Pisqorky.mouse_input")

        chosen = None
        while chosen is None:
            event = pygame.event.wait()
            if event.type == pygame.MOUSEBUTTONDOWN:
                chosen = self.mouse_click_to_action(event.pos)
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

    _A = 40


# Register the Pisqorky game.
BoardGame.register_game("pisqorky", Pisqorky)
