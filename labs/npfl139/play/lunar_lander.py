#!/usr/bin/env python3
import argparse

import gymnasium as gym
import gymnasium.utils.play
import pygame

parser = argparse.ArgumentParser()
parser.add_argument("--fps", default=25, type=int, help="Frames per second.")
parser.add_argument("--zoom", default=2.0, type=float, help="Zoom ratio.")


class ReturnReporter():
    def __init__(self):
        self.rewards = 0

    def __call__(self, obs_t, obs_tp1, action, rew, terminated, truncated, info):
        self.rewards += rew
        if terminated or truncated:
            print("Episode reward:", self.rewards)
            self.rewards = 0


def main(args: argparse.Namespace) -> None:
    gym.utils.play.play(
        gym.make("LunarLander-v3", render_mode="rgb_array"),
        callback=ReturnReporter(), fps=args.fps, zoom=args.zoom, noop=0, keys_to_action={
            (pygame.K_UP,): 2,
            (pygame.K_LEFT,): 1,
            (pygame.K_LEFT, pygame.K_UP): 1,
            (pygame.K_RIGHT,): 3,
            (pygame.K_RIGHT, pygame.K_UP): 3,
        })


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
