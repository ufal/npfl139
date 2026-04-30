# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from .az_quiz import AZQuiz
from .board_game_player import BoardGamePlayer


class AZQuizPlayerMouse(BoardGamePlayer[AZQuiz]):
    """An interactive player using mouse to select an action."""
    def play(self, game: AZQuiz) -> int:
        print(f"Choose action for player {game.to_play}: ", end="", flush=True)
        action = game.mouse_input()
        print(f"action {action}", flush=True)
        return action


AZQuiz.register_player("mouse", AZQuizPlayerMouse)
