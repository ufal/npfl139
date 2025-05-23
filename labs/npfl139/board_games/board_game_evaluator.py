#!/usr/bin/env python3
#
# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import time

from .board_game import BoardGame
from .board_game_player import BoardGamePlayer


def evaluate(
    game_class: type[BoardGame], players: tuple[BoardGamePlayer, BoardGamePlayer],
    games: int, first_chosen: bool, render: bool = False, verbose: bool = False,
) -> float:
    if first_chosen:
        if game_class.__name__.startswith("AZQuiz"):
            assert games % game_class.ACTIONS == 0, \
                "If `first_chosen` is True, the number of games must be divisble by the number of actions"
            def first_move_selector(game_index: int) -> int:  # noqa: E301
                return game_index % game_class.ACTIONS

        elif game_class.__name__ == "Pisqorky":
            assert games % 50 == 0, \
                "If `first_chosen` is True, the number of games must be divisible by 50"
            def first_move_selector(index: int) -> int:  # noqa: E301
                return 112 if index % 50 == 49 else (index % 50 // 7 + 4) * 15 + (index % 50 % 7 + 4)

        else:
            raise ValueError(f"The game {game_class.__name__} does not support first move selection")

    wins = [0, 0]
    for i in range(games):
        for to_start in range(2):
            game = game_class()
            if first_chosen:
                game.move(first_move_selector(i))
            while not game.outcome(game.to_play):
                game.move(players[to_start ^ game.to_play].play(game.clone()))
                if render:
                    game.render()
                    time.sleep(0.2)
            if game.outcome(to_start) == game.Outcome.WIN:
                wins[to_start] += 1
            if render:
                time.sleep(2.0)

        if verbose:
            g = i + 1
            print("First player win rate after {} games: {:.2f}% ({:.2f}% and {:.2f}% when starting and not starting)"
                  .format(2 * g, 100 * (wins[0] + wins[1]) / (2 * g), 100 * wins[0] / g, 100 * wins[1] / g))

    return (wins[0] + wins[1]) / (2 * games)
