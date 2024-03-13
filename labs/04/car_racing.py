#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch

import car_racing_environment
import wrappers
car_racing_environment.register()

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--continuous", default=1, type=int, help="Use continuous actions.")
parser.add_argument("--frame_skip", default=1, type=int, help="Frame skip.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent

        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO: Choose a greedy action
                action = ...
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Implement a suitable RL algorithm and train the agent.
    #
    # If you want to create N multiprocessing parallel environments, use
    #   vector_env = gym.make_vec("CarRacingFS-v2", N, gym.VectorizeMode.ASYNC,
    #                             frame_skip=args.frame_skip, continuous=args.continuous)
    #   vector_env.reset(seed=args.seed)  # The individual environments get incremental seeds
    #
    # See https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.0.0a1 for the
    # description of the new autoreset behavior - beware that it is different from 0.29.0.
    training = True
    while training:
        ...


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        gym.make("CarRacingFS-v2", frame_skip=args.frame_skip, continuous=args.continuous), args.seed, args.render_each,
        evaluate_for=15, report_each=1)

    main(env, args)
