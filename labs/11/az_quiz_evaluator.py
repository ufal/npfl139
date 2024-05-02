#!/usr/bin/env python3
import argparse
import importlib
import os
import time

import az_quiz

import numpy as np


def load_player(args: argparse.Namespace, player: str):
    player, *player_args = player.split(":")

    if player.endswith(".py"):
        player = player[:-3]

    def loader():
        module = importlib.import_module(player)
        module_args = module.parser.parse_args(player_args)
        module_args.recodex = True
        if hasattr(module_args, "seed") and module_args.seed is None and args.seed is not None:
            module_args.seed = args.seed
        try:
            cwd = os.getcwd()
            os.chdir(os.path.dirname(module.__file__))
            return module.main(module_args)
        finally:
            os.chdir(cwd)

    if args.multiprocessing:
        import multiprocessing

        class Player:
            def __init__(self):
                self._conn, child_conn = multiprocessing.Pipe()
                self._p = multiprocessing.Process(target=Player._worker, args=(child_conn, loader), daemon=True)
                self._p.start()
            def _worker(conn, loader):
                player = loader()
                while True:
                    conn.send(player.play(conn.recv()))
            def play(self, game):
                self._conn.send(game)
                return self._conn.recv()
        return Player()

    else:
        return loader()


def evaluate(
    players: list, games: int, randomized: bool, first_chosen: bool, render: bool = False, verbose: bool = False
) -> float:
    assert not first_chosen or games % az_quiz.AZQuiz.ACTIONS == 0, \
        "If `first_chosen` is True, the number of games must be divisble by the number of actions"

    wins = [0, 0]
    for i in range(games):
        for to_start in range(2):
            game = az_quiz.AZQuiz(randomized)
            if first_chosen:
                game.move(i % game.ACTIONS)
            while game.winner is None:
                game.move(players[to_start ^ game.to_play].play(game.clone()))
                if render:
                    game.render()
                    time.sleep(0.3)
            if game.winner == to_start:
                wins[to_start] += 1
            if render:
                time.sleep(1.0)

        if verbose:
            g = i + 1
            print("First player win rate after {} games: {:.2f}% ({:.2f}% and {:.2f}% when starting and not starting)"
                  .format(2 * g, 100 * (wins[0] + wins[1]) / (2 * g), 100 * wins[0] / g, 100 * wins[1] / g))

    return (wins[0] + wins[1]) / (2 * games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("player_1", type=str, help="First player module")
    parser.add_argument("player_2", type=str, help="Second player module")
    parser.add_argument("--first_chosen", default=False, action="store_true", help="The first move is chosen")
    parser.add_argument("--games", default=56, type=int, help="Number of alternating games to evaluate")
    parser.add_argument("--multiprocessing", default=False, action="store_true", help="Load players in sep. processes")
    parser.add_argument("--randomized", default=False, action="store_true", help="Can answering produce random result")
    parser.add_argument("--render", default=False, action="store_true", help="Should the games be rendered")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    evaluate(
        [load_player(args, args.player_1), load_player(args, args.player_2)],
        games=args.games,
        randomized=args.randomized,
        first_chosen=args.first_chosen,
        render=args.render,
        verbose=True,
    )
