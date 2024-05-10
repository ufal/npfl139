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
        """Return the winner of the game (0/1), 2 on draw, `None` if the game is not over."""

    def valid(self, action):
        """Return whether the given action is valid."""

    def valid_actions(self):
        """Return the list of valid actions."""

    def move(self, action):
        """Execute the given action."""


class Pisqorky(BoardGame):
    ACTIONS = 225
    N = 15
    C = 3

    def __init__(self, randomized=False):
        self._board = np.zeros([self.N, self.N], dtype=np.int8)
        self._to_play = 0
        self._winner = None
        self._screen = None

    def clone(self, swap_players=False):
        clone = Pisqorky()
        if swap_players:
            clone._board[:, :] = (self._board == 2) + 2 * (self._board == 1)
            clone._to_play = 1 - self._to_play
            clone._winner = 1 - self._winner if self._winner is not None else None
        else:
            clone._board[:, :] = self._board
            clone._to_play = self._to_play
            clone._winner = self._winner
        return clone

    @property
    def board(self):
        return np.eye(3, dtype=np.uint8)[self._board]

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
            and self._board[action // self.N, action % self.N] == 0

    def valid_actions(self):
        return np.nonzero(self._board.ravel() == 0)[0] if self._winner is None else []

    def move(self, action):
        if not self.valid(action):
            raise ValueError("An invalid action to Pisqorky.move")
        self._board[action // self.N, action % self.N] = self._to_play + 1
        self._to_play = 1 - self._to_play

        # Check for a winner
        free_fields = False
        for y in range(self.N):
            for x in range(self.N):
                if self._board[y, x] == 0:
                    free_fields = True
                    continue
                field = self._board[y, x]
                if ((x >= 4 and y + 4 < self.N and field ==
                     self._board[y + 1, x - 1] == self._board[y + 2, x - 2] ==
                     self._board[y + 3, x - 3] == self._board[y + 4, x - 4]) or \
                    (y + 4 < self.N and field ==
                     self._board[y + 1, x] == self._board[y + 2, x] ==
                     self._board[y + 3, x] == self._board[y + 4, x]) or \
                    (x + 4 < self.N and y + 4 < self.N and field ==
                     self._board[y + 1, x + 1] == self._board[y + 2, x + 2] ==
                     self._board[y + 3, x + 3] == self._board[y + 4, x + 4]) or \
                    (x + 4 < self.N and field ==
                     self._board[y, x + 1] == self._board[y, x + 2] ==
                     self._board[y, x + 3] == self._board[y, x + 4])):
                    self._winner = field - 1
                    return
        if not free_fields:
            self._winner = 2

    def render(self):
        import pygame
        A, W = 40, self.N * 40
        white, black, red, blue = (255, 255, 255), (0, 0, 0), (224, 32, 32), (32, 32, 224)

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
        if self._winner is not None:
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

        return chosen
