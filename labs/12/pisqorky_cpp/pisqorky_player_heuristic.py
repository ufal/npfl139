#!/usr/bin/env python3
import argparse

import numpy as np

import pisqorky
import pisqorky_cpp

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.


class Player:
    def play(self, pisqorky):
        return pisqorky_cpp.heuristic(pisqorky.board_internal, pisqorky.to_play)


def main(args):
    return Player()
