# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np

from .az_quiz import AZQuiz
from .board_game_player import BoardGamePlayer


class AZQuizPlayerSimpleHeuristic(BoardGamePlayer[AZQuiz]):
    """A simple heuristic for AZQuiz."""
    CENTER = 12
    ANCHORS = [4, 16, 19]

    def __init__(self, seed: int | None = None):
        super().__init__()
        self._generator = np.random.RandomState(seed)

    def play(self, game: AZQuiz) -> int:
        if game.valid(self.CENTER):
            return self.CENTER

        any_anchor = any(map(game.valid, self.ANCHORS))

        action = None
        while action is None or not game.valid(action):
            if any_anchor:
                action = self._generator.choice(self.ANCHORS)
            else:
                action = self._generator.randint(game.ACTIONS)

        return action


AZQuiz.register_player("simple_heuristic", AZQuizPlayerSimpleHeuristic)
