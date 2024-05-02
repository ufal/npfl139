#!/usr/bin/env python3
import argparse
from typing import Optional

import numpy as np

import az_quiz

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.


class Player:
    def __init__(self, seed: Optional[int] = None):
        self._generator = np.random.RandomState(seed)

    def play(self, az_quiz):
        action = None
        while action is None or not az_quiz.valid(action):
            action = self._generator.randint(az_quiz.ACTIONS)

        return action


def main(args):
    return Player(seed=args.seed)
