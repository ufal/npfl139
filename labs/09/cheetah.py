#!/usr/bin/env python3
import argparse
import collections

import gymnasium as gym
import numpy as np
import torch

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="HalfCheetah-v5", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Evaluation in ReCodEx
    if args.recodex:
        # TODO: Load a pretrain model and perform evaluation.
        ...
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO: Predict the action using the greedy policy.
                action = ...
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Perform training
    raise NotImplementedError()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
