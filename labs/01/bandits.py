#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use or 0 for averaging.")
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
parser.add_argument("--episodes", default=100, type=int, help="Episodes to perform.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
parser.add_argument("--initial", default=0, type=float, help="Initial estimation of values.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.


# A class providing MultiArmedBandits environment.
class MultiArmedBandits():
    def __init__(self, bandits: int, seed: int) -> None:
        self.__generator = np.random.RandomState(seed)
        self.__bandits = [None] * bandits
        self.reset()

    def reset(self) -> None:
        for i in range(len(self.__bandits)):
            self.__bandits[i] = self.__generator.normal(0., 1.)

    def step(self, action: int) -> float:
        return self.__generator.normal(self.__bandits[action], 1.)

    def greedy(self, epsilon: float) -> bool:
        return self.__generator.uniform() >= epsilon


def main(env: MultiArmedBandits, args: argparse.Namespace) -> float:
    # TODO: Initialize the estimates for all bandits, to `args.initial`.

    rewards = 0
    for step in range(args.episode_length):
        # TODO: Select either a greedy action (if `env.greedy(args.epsilon)` is True)
        # or uniformly random action (otherwise).
        action = ...

        # Perform the action.
        reward = env.step(action)
        rewards += reward

        # TODO: Update parameters, either using averaging (when `args.alpha` == 0)
        # or by an update with a learning rate of `args.alpha`.
        ...

    return rewards / args.episode_length


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = MultiArmedBandits(args.bandits, seed=args.seed)

    # Set random seed
    np.random.seed(args.seed)

    returns = []
    for _ in range(args.episodes):
        returns.append(main(env, args))

    # Print the mean and std
    print("{:.2f} {:.2f}".format(np.mean(returns), np.std(returns)))
