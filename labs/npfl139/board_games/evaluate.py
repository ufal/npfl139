#!/usr/bin/env python3
#
# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import argparse
import importlib
import multiprocessing
import pathlib
import os
import signal
import sys

from .board_game import BoardGame
from .board_game_evaluator import evaluate


def load_player(args: argparse.Namespace, player: str):
    player, *player_args = player.split(":")

    # Try a reference player first
    try:
        return BoardGame.from_name(args.game).player_from_name(player)(args.seed)
    except Exception:
        pass

    # Otherwise, load a player from a module in a separate process
    if player.endswith(".py"):
        player = player[:-3]

    class MultiprocessingPlayer:
        def __init__(self):
            self._conn, child_conn = multiprocessing.Pipe()
            self._p = multiprocessing.Process(target=self._worker, args=(child_conn,), daemon=True)
            self._p.start()
            assert self._conn.recv() == "ready"

        @staticmethod
        def _worker(conn):
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                module_path = str(pathlib.Path(*player.split(".")).parent.resolve())
                sys.path.insert(0, module_path)
                os.chdir(module_path)

                module = importlib.import_module(player)
                module_args = module.parser.parse_args(player_args)
                module_args.recodex = True
                if hasattr(module_args, "seed") and module_args.seed is None and args.seed is not None:
                    module_args.seed = args.seed
                player_interface = module.main(module_args)

                conn.send("ready")
                while True:
                    conn.send(player_interface.play(conn.recv()))
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                sys.stderr.flush()
                os.kill(os.getppid(), signal.SIGTERM)

        def play(self, game):
            self._conn.send(game)
            return self._conn.recv()

    return MultiprocessingPlayer()


if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("player_1", type=str, help="First player module")
    parser.add_argument("player_2", type=str, help="Second player module")
    parser.add_argument("--first_chosen", default=False, action="store_true", help="The first move is chosen")
    parser.add_argument("--game", default="az_quiz", type=str, help="Game to evaluate")
    parser.add_argument("--games", default=56, type=int, help="Number of alternating games to evaluate")
    parser.add_argument("--render", default=False, action="store_true", help="Should the games be rendered")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    evaluate(
        game_class=BoardGame.from_name(args.game),
        players=(load_player(args, args.player_1), load_player(args, args.player_2)),
        games=args.games,
        first_chosen=args.first_chosen,
        render=args.render,
        verbose=True,
    )
