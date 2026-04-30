# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from .az_quiz import AZQuiz
from .board_game_player import BoardGamePlayer


class AZQuizPlayerKeyboard(BoardGamePlayer[AZQuiz]):
    """An interactive player using keyboard to select an action."""
    def play(self, game: AZQuiz) -> int:
        return game.keyboard_input()


AZQuiz.register_player("keyboard", AZQuizPlayerKeyboard)
