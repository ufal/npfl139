#!/usr/bin/env python3
import argparse

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
    def play(self, az_quiz):
        print("Choose action for player {}: ".format(az_quiz.to_play), end="", flush=True)
        action = az_quiz.mouse_input()
        print("action {}".format(action), flush=True)

        return action


def main(args):
    return Player()
