#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> np.ndarray:
    # Create the environment
    env = gym.make("FrozenLake-v1")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    # Set random seed
    np.random.seed(args.seed)

    # Behaviour policy is uniformly random.
    # Target policy uniformly chooses either action 1 or 2.
    V = np.zeros(env.observation_space.n)
    C = np.zeros(env.observation_space.n)

    for _ in range(args.episodes):
        state, done = env.reset()[0], False

        # Generate episode
        episode = []
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state

        # TODO: Update V using weighted importance sampling.

    return V


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    value_function = main(args)

    # Print the final value function V
    for row in value_function.reshape(4, 4):
        print(" ".join(["{:5.2f}".format(x) for x in row]))
