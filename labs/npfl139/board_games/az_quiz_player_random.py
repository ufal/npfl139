# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np

from .az_quiz import AZQuiz
from .board_game_player import BoardGamePlayer


class AZQuizPlayerRandom(BoardGamePlayer[AZQuiz]):
    """Player playing completely randomly."""
    def __init__(self, seed: int | None = None):
        self._generator = np.random.RandomState(seed)

    def play(self, game: AZQuiz) -> int:
        action = None
        while action is None or not game.valid(action):
            action = self._generator.randint(game.ACTIONS)

        return action


AZQuiz.register_player("random", AZQuizPlayerRandom)
