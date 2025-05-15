# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from .board_game import BoardGame
from .board_game_player import BoardGamePlayer
from .board_game_evaluator import evaluate

from .az_quiz import AZQuiz, AZQuizRandomized
from .az_quiz_player_fork_heuristic import AZQuizPlayerForkHeuristic
from .az_quiz_player_keyboard import AZQuizPlayerKeyboard
from .az_quiz_player_mouse import AZQuizPlayerMouse
from .az_quiz_player_random import AZQuizPlayerRandom
from .az_quiz_player_simple_heuristic import AZQuizPlayerSimpleHeuristic

from .pisqorky import Pisqorky
from .pisqorky_player_heuristic import PisqorkyPlayerHeuristic
from .pisqorky_player_mouse import PisqorkyPlayerMouse
from .pisqorky_player_random import PisqorkyPlayerRandom
