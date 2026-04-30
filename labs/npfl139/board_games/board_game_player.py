# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Generic, TypeVar

import numpy as np

BoardGameT = TypeVar("BoardGameT")  # an instance of a board game


class BoardGamePlayer(Generic[BoardGameT]):
    def __init__(self, seed: int | None = None):
        pass

    def play(self, game: BoardGameT) -> int:
        """Return an action to play."""
        raise NotImplementedError()

    def evaluate(self, game: BoardGameT) -> tuple[np.ndarray, float]:
        """An optional method evaluating the game state and returning the predicted policy and value.

        Returns:
          (policy, value): The `policy` is a distribution over the valid actions; the `value`
            is the current value estimate scaled to [0, 1] range.
        """
        raise NotImplementedError()

    def evaluate_values(self, game: BoardGameT) -> np.ndarray:
        """An optional method evaluating the value of all actions.

        Returns:
          values: The values of all actions scaled to [0, 1] range; arbitrary for invalid actions.
        """
        raise NotImplementedError()

    def evaluate_mcts(self, game: BoardGame, simulations: int) -> np.ndarray:
        """An optional method evaluating the policy using MCTS.

        Returns:
          policy: The policy distribution over the valid actions.
        """
        raise NotImplementedError()
