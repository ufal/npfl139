#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=..., type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=..., type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=..., type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=..., type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=..., type=float, help="Learning rate.")


class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Similarly to reinforce with baseline, define two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # Use independent networks for both of them, each with
        # `args.hidden_layer_size` neurons in one ReLU hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        raise NotImplementedError()

    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Train the policy network using policy gradient theorem
        # and the value network using MSE.
        #
        # The `args.entropy_regularization` might be used to include actor
        # entropy regularization -- the assignment can be solved even without
        # it, but my reference solution learns more quickly when using it.
        # In any case, `torch.distributions.Categorical` is the suitable distribution
        # offering the `.entropy()` method.
        raise NotImplementedError()

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        raise NotImplementedError()

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return estimates of value function.
        raise NotImplementedError()


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # TODO: Predict the action using the greedy policy.
            action = ...
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC)
    states = vector_env.reset(seed=args.seed)[0]

    training, autoreset = True, np.zeros(args.envs, dtype=bool)
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Choose actions using `network.predict_actions`.
            actions = ...

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated

            # TODO: Compute estimates of returns by one-step bootstrapping

            # TODO: Train network using current states, chosen actions and estimated returns.
            # However, note that when `autoreset[i] == True`, the `i`-th environment has
            # just reset, so `states[i]` is the terminal state of a previous episode
            # and `nextstate` is the initial state of a new episode.

            states = next_states
            autoreset = dones

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
