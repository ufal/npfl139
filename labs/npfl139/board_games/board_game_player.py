# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Generic, TypeVar

import numpy as np

BoardGameT = TypeVar("BoardGameT")  # An instance of a board game


class BoardGamePlayer(Generic[BoardGameT]):
    def __init__(self, seed: int | None = None):
        pass

    def play(self, game: BoardGameT) -> int:
        """Return an action to play."""
        raise NotImplementedError()

    def evaluate(self, game: BoardGameT) -> tuple[np.ndarray, dict[int, float]]:
        """An optional method evaluating the game state and returning predicted policy and chosen value estimates.

        Returns:
          (policy, value): The `policy` is a distribution over valid actions; the `value`
            is a dictionary assigning value estimate to chosen actions (possibly empty).
        """
        raise NotImplementedError()
