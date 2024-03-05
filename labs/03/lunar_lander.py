#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=..., type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=..., type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=..., type=float, help="Discounting factor.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    np.random.seed(args.seed)

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
    training = True
    while training:
        # To generate an expert trajectory, you can use the following:
        #   trajectory = env.expert_trajectory()

        # Otherwise, you can generate a training episode the usual way:
        state, done = env.reset()[0], False
        while not done:
            action = ...
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ...


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, args.render_each)

    main(env, args)
