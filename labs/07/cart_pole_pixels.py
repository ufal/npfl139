#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.7")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        ...

        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            while not done:
                # TODO: Choose a greedy action.
                action = ...
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Perform training
    raise NotImplementedError()


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
