# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np

from .board_game import BoardGame
from .board_game_player import BoardGamePlayer


class AZQuiz(BoardGame):
    ACTIONS: int = 28
    """Number of actions in the game."""
    N: int = 7
    """The number of rows and columns in the game board."""
    C: int = 4
    """The number of features in the board representation."""

    def __init__(self, randomized=False):
        self._board = np.tri(self.N, dtype=np.int8) - 1
        self._randomized = randomized
        self._to_play = 0
        self._outcome = None
        self._screen = None
        self._last_action, self._winning_stones = None, None

    def clone(self, swap_players=False) -> "AZQuiz":
        clone = AZQuiz(self._randomized)
        if swap_players:
            clone._board = self._SWAP_PLAYERS[self._board + 1]
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
        return self._REPRESENTATION[self._board + 1]

    @property
    def to_play(self) -> int:
        return self._to_play

    def outcome(self, player: int) -> BoardGame.Outcome | None:
        return self._outcome if self._outcome is None or player == self._to_play else self._outcome.reverse()

    def valid(self, action: int) -> bool:
        return self._outcome is None and action >= 0 and action < self.ACTIONS \
            and self._board[self._ACTION_Y[action], self._ACTION_X[action]] < 2

    def valid_actions(self) -> list[int]:
        return np.nonzero(self._board[self._ACTION_Y, self._ACTION_X] < 2)[0] if self._outcome is None else []

    def move(self, action: int):
        self._last_action = action
        return self._move(action, np.random.uniform() if self._randomized else 0)

    def all_moves(self, action):
        success = self.clone()
        success._move(action, 0.)
        if not self._randomized:
            return [(1.0, success)]

        failure = self.clone()
        failure._move(action, 1.)
        if self._board[self._ACTION_Y[action], self._ACTION_X[action]] == 0:
            success_probability = self._INITIAL_QUESTION_PROB
        else:
            success_probability = self._ADDITIONAL_QUESTION_PROB
        return [(success_probability, success), (1. - success_probability, failure)]

    def _move(self, action, random_value):
        if not self.valid(action):
            raise ValueError("An invalid action to AZQuiz.move")

        if self._board[self._ACTION_Y[action], self._ACTION_X[action]] == 0:
            if random_value <= self._INITIAL_QUESTION_PROB:
                self._board[self._ACTION_Y[action], self._ACTION_X[action]] = 2 + self._to_play
            else:
                self._board[self._ACTION_Y[action], self._ACTION_X[action]] = 1
        else:
            if random_value > self._ADDITIONAL_QUESTION_PROB:
                self._to_play = 1 - self._to_play
            self._board[self._ACTION_Y[action], self._ACTION_X[action]] = 2 + self._to_play
        self._to_play = 1 - self._to_play

        edges, visited = np.zeros(2, dtype=np.uint8), np.zeros([self.N, self.N], dtype=np.uint8)
        for j in range(self.N):
            edges[:] = False
            field = self._board[j, 0]
            if field >= 2:
                self._traverse(j, 0, field, edges, visited)
                if edges.all():
                    self._outcome = self.Outcome.WIN if field - 2 == self._to_play else self.Outcome.LOSS
                    self._winning_stones = visited == 1
                visited += visited > 0

    def _traverse(self, j, i, field, edges, visited):
        if visited[j, i]: return
        visited[j, i] = 1

        if j == i: edges[0] = True
        if j == self.N - 1: edges[1] = True
        if j - 1 >= 0:
            if i - 1 >= 0 and self._board[j - 1, i - 1] == field: self._traverse(j - 1, i - 1, field, edges, visited)
            if self._board[j - 1, i] == field: self._traverse(j - 1, i, field, edges, visited)
        if i - 1 >= 0 and self._board[j, i - 1] == field: self._traverse(j, i - 1, field, edges, visited)
        if i + 1 < self.N and self._board[j, i + 1] == field: self._traverse(j, i + 1, field, edges, visited)
        if j + 1 < self.N:
            if self._board[j + 1, i] == field: self._traverse(j + 1, i, field, edges, visited)
            if i + 1 < self.N and self._board[j + 1, i + 1] == field: self._traverse(j + 1, i + 1, field, edges, visited)

    def _render(self):
        A = 40
        I = 35
        W = A * 13
        COLORS = np.array([[210, 210, 210], [58, 58, 58], [58, 147, 192], [254, 147, 17],
                           [158, 158, 158], [77, 195, 255], [255, 194, 77]], dtype=np.uint8)

        image = np.zeros([W, W, 3], dtype=np.uint8)
        for j in range(self.N):
            for i in range(j + 1):
                x = int((i - j/2) * 2 * 0.866 * A + W/2)
                y = int((j - (self.N - 1)/2) * 1.5 * A + W/2)
                for yo in range(-I, I + 1):
                    xo = int(min(2 * 0.866 * (I - abs(yo)), 0.866 * I))
                    image[x - xo:x + xo + 1, y + yo] = COLORS[self._board[j, i] + 3 * (
                        (j * (j + 1) // 2 + i == self._last_action) or
                        (self._winning_stones is not None and self._winning_stones[j, i]))]

        return image

    def render(self):
        import pygame

        image = self._render()

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.get_surface() or pygame.display.set_mode(image.shape[:2])
            self._screen_surface = pygame.Surface(image.shape[:2])

        pygame.pixelcopy.array_to_surface(self._screen_surface, image)
        self._screen.blit(self._screen_surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()

    def mouse_input(self):
        import pygame

        centers, A, I, W = [], 40, 35, 40 * 13
        for j in range(self.N):
            for i in range(j + 1):
                centers.append(((i - j/2) * 2 * 0.866 * A + W/2,
                                (j - (self.N - 1)/2) * 1.5 * A + W/2))

        self.render()
        chosen = None
        while chosen is None:
            event = pygame.event.wait()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for action, (x, y) in enumerate(centers):
                    distance2 = (event.pos[0] - x)**2 + (event.pos[1] - y)**2
                    if distance2 < I * I and self.valid(action):
                        chosen = action
                        break
            if event.type == pygame.WINDOWEXPOSED:
                self._screen.blit(self._screen_surface, (0, 0))
                pygame.display.flip()
            if event.type == pygame.QUIT:
                print("Window closed, stopping application.")
                exit()

        return chosen

    def keyboard_input(self):
        SYMBOLS = [".", "*", "O", "X"]

        board, action = [], 0
        for j in range(self.N):
            board.append("")
            for mode in range(2):
                board[-1] += "  " * (self.N - 1 - j)
                for i in range(j + 1):
                    board[-1] += " " + (SYMBOLS[self._board[j, i]] * 2 if mode == 0 else "{:2d}".format(action + i)) + " "
                board[-1] += "  " * (self.N - 1 - j)
            action += j + 1

        print("\n".join(board), flush=True)

        action = None
        while action is None or not self.valid(action):
            try:
                action = int(input("Choose action for player {}: ".format(SYMBOLS[2 + self.to_play])))
            except ValueError:
                pass
        return action

    @staticmethod
    def player_from_name(player_name: str) -> type[BoardGamePlayer["AZQuiz"]]:
        assert player_name in AZQuiz._players, f"AZQuiz.player_from_name got unknown player name: {player_name}"
        return AZQuiz._players[player_name]

    @staticmethod
    def register_player(player_name: str, player: type[BoardGamePlayer["AZQuiz"]]):
        AZQuiz._players[player_name] = player

    _players: dict[str, type[BoardGamePlayer["AZQuiz"]]] = {}

    _INITIAL_QUESTION_PROB = 0.8
    _ADDITIONAL_QUESTION_PROB = 0.7

    _ACTION_Y = np.array([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6], dtype=np.int8)
    _ACTION_X = np.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6], dtype=np.int8)
    _REPRESENTATION = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,1], [1,0,0,1], [0,1,0,1]], dtype=np.uint8)
    _SWAP_PLAYERS = np.array([-1,0,1,3,2])


# Explicit class for the randomized version of the game.
class AZQuizRandomized(AZQuiz):
    def __init__(self):
        super().__init__(randomized=True)


# Register both AZQuiz variants.
BoardGame.register_game("az_quiz", AZQuiz)
BoardGame.register_game("az_quiz_randomized", AZQuizRandomized)
