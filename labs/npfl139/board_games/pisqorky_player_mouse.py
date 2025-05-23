# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from .pisqorky import Pisqorky
from .board_game_player import BoardGamePlayer


class PisqorkyPlayerMouse(BoardGamePlayer[Pisqorky]):
    """An interactive player using mouse to select an action."""
    def play(self, game: Pisqorky) -> int:
        print("Choose action for player {}: ".format(game.to_play), end="", flush=True)
        action = game.mouse_input()
        print("action {}".format(action), flush=True)
        return action


Pisqorky.register_player("mouse", PisqorkyPlayerMouse)
