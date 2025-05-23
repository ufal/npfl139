# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np

from .pisqorky import Pisqorky
from .board_game_player import BoardGamePlayer


class PisqorkyPlayerHeuristic(BoardGamePlayer[Pisqorky]):
    """A moderately strong greedy heuristic player for Pisqorky."""
    def __init__(self, seed: int | None = None):
        self._generator = np.random.RandomState(seed)

        # Precompute the coordinates of all possible lines of 5 stones.
        lines_x, lines_y = [], []
        for dy, dx in zip([1, 1, 1, 0], [-1, 0, 1, 1]):
            for shift in range(-4, 1):
                lines_y.append(dy * (np.arange(5) + shift))
                lines_x.append(dx * (np.arange(5) + shift))
        self.lines_x = np.array(lines_x, dtype=np.int32)
        self.lines_y = np.array(lines_y, dtype=np.int32)

        # The coefficients for the heuristic evaluation function.
        self.coefs = np.array([[0, 5, 25, 125, 1000], [0, 4, 20, 100, 400]])

    def play(self, game: Pisqorky) -> int:
        board = (game.board == game.to_play + 1) + ((game.board == 2 - game.to_play) << 3)

        empty_board = (board == 0)
        if np.all(empty_board):
            # First move
            y = Pisqorky.N // 2 + self._generator.randint(-2, 3)
            x = Pisqorky.N // 2 + self._generator.randint(-2, 3)
            return y * Pisqorky.N + x

        best_score, best_moves = -1, []
        for y, x in np.argwhere(empty_board):
            if np.all(board[max(0, y - 2):min(Pisqorky.N, y + 3), max(0, x - 2):min(Pisqorky.N, x + 3)] == 0):
                continue
            coords_y, coords_x = y + self.lines_y, x + self.lines_x
            valids = np.all((coords_x >= 0) & (coords_x < Pisqorky.N) & (coords_y >= 0) & (coords_y < Pisqorky.N), axis=1)
            coords_y, coords_x = coords_y[valids], coords_x[valids]

            line_pieces = board[coords_y, coords_x].sum(axis=1)
            score = self.coefs[0, line_pieces[(line_pieces & 0b111000) == 0]].sum() + \
                self.coefs[1, line_pieces[(line_pieces & 0b000111) == 0] >> 3].sum()

            if score > best_score:
                best_score = score
                best_moves = [y * Pisqorky.N + x]
            elif score == best_score:
                best_moves.append(y * Pisqorky.N + x)

        return self._generator.choice(best_moves)


Pisqorky.register_player("heuristic", PisqorkyPlayerHeuristic)
