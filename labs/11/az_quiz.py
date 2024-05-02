#!/usr/bin/env python3
import sys

import numpy as np


class BoardGame:
    ACTIONS: int
    N: int
    C: int

    def clone(self, swap_players=False):
        """Clone the game state, optionally swapping the players."""

    @property
    def board(self):
        """Return the board as a NxNxC numpy array of features."""

    @property
    def board_internal(self):
        """Return the internal representation of board as a NxN numpy array."""

    @property
    def to_play(self):
        """Return the current player."""

    @property
    def winner(self):
        """Return the winner of the game, or `None` if the game is not over."""

    def valid(self, action):
        """Return whether the given action is valid."""

    def valid_actions(self):
        """Return the list of valid actions."""

    def move(self, action):
        """Execute the given action."""


class AZQuiz(BoardGame):
    ACTIONS = 28
    N = 7
    C = 4

    def __init__(self, randomized=False):
        self._board = np.tri(self.N, dtype=np.int8) - 1
        self._randomized = randomized
        self._to_play = 0
        self._winner = None
        self._screen = None

    def clone(self, swap_players=False):
        clone = AZQuiz(self._randomized)
        if swap_players:
            clone._board = self._SWAP_PLAYERS[self._board + 1]
            clone._to_play = 1 - self._to_play
            clone._winner = 1 - self._winner if self._winner is not None else None
        else:
            clone._board[:, :] = self._board
            clone._to_play = self._to_play
            clone._winner = self._winner
        return clone

    @property
    def board(self):
        return self._REPRESENTATION[self._board + 1]

    @property
    def board_internal(self):
        return self._board

    @property
    def to_play(self):
        return self._to_play

    @property
    def winner(self):
        return self._winner

    def valid(self, action):
        return self._winner is None and action >= 0 and action < self.ACTIONS \
            and self._board[self._ACTION_Y[action], self._ACTION_X[action]] < 2

    def valid_actions(self):
        return np.nonzero(self._board[self._ACTION_Y, self._ACTION_X] < 2)[0] if self._winner is None else []

    def move(self, action):
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
                    self._winner = field - 2

    def _traverse(self, j, i, field, edges, visited):
        if visited[j, i]: return
        visited[j, i] = True

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
        COLORS = np.array([[210, 210, 210], [58, 58, 58], [58, 147, 192], [254, 147, 17]], dtype=np.uint8)

        image = np.zeros([W, W, 3], dtype=np.uint8)
        for j in range(self.N):
            for i in range(j + 1):
                x = int((i - j/2) * 2 * 0.866 * A + W/2)
                y = int((j - (self.N - 1)/2) * 1.5 * A + W/2)
                for yo in range(-I, I + 1):
                    xo = int(min(2 * 0.866 * (I - abs(yo)), 0.866 * I))
                    image[x - xo:x + xo + 1, y + yo] = COLORS[self._board[j, i]]

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

    _INITIAL_QUESTION_PROB = 0.8
    _ADDITIONAL_QUESTION_PROB = 0.7

    _ACTION_Y = np.array([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6], dtype=np.int8)
    _ACTION_X = np.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6], dtype=np.int8)
    _REPRESENTATION = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,1], [1,0,0,1], [0,1,0,1]], dtype=np.uint8)
    _SWAP_PLAYERS = np.array([-1,0,1,3,2])
