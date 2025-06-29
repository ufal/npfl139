#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.10")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="SingleCollect-v0", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=..., type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=..., type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=..., type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=..., type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=..., type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=..., type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=..., type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=..., type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=..., type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=..., type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=..., type=int, help="Steps for each worker to perform.")


class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self._args = args

        # TODO: Create an actor using a single hidden layer with `args.hidden_layer_size`
        # units and ReLU activation, produce a policy with `env.action_space.n` discrete actions.
        self._actor = ...

        # TODO: Create a critic (value predictor) consisting of a single hidden layer with
        # `args.hidden_layer_size` units and ReLU activation, and an output layer with a single output.
        self._critic = ...

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, action_probs: torch.Tensor,
              advantages: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform a single training step of the PPO algorithm.
        # For the policy model, the sum is the sum of:
        # - the PPO loss, where `self._args.clip_epsilon` is used to clip the probability ratio
        # - the entropy regularization with coefficient `self._args.entropy_regularization`.
        #   You can compute it for example using the `torch.distributions.Categorical` class.
        ...

        # TODO: The critic model is trained in a standard way, by using the MSE
        # error between the predicted value function and target returns.
        ...

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        raise NotImplementedError()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return estimates of value function.
        raise NotImplementedError()

    # Serialization methods.
    def save_actor(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action by using a greedy policy.
            action = ...
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create an asynchronous vector environment for training.
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})

    # Training
    state = vector_env.reset(seed=args.seed)[0]
    training, iteration = True, 0
    while training:
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[args.worker_steps, args.envs]`.
        states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(args.worker_steps):
            # TODO: Choose `action`, which is a vector of `args.envs` actions, each
            # sampled from the corresponding policy generated by the `agent.predict`
            # executed on the vector `state`.
            action = ...

            # Perform the environment interaction.
            next_state, reward, terminated, truncated, _ = vector_env.step(action)
            done = terminated | truncated

            # TODO: Compute and collect the required quantities.
            ...

            state = next_state

        # TODO: Estimate `advantages` and `returns` (they differ only by the value function estimate)
        # using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
        # You need to handle both the cases that (a) the last episode is probably unfinished, and
        # (b) there are multiple episodes in the collected data.
        advantages, returns = ...

        # TODO: Train for `args.epochs` using the collected data. In every epoch,
        # you should randomly sample batches of size `args.batch_size` from the collected data.
        # A possible approach is to create a dataset of `(states, actions, action_probs, advantages, returns)`
        # quintuples using a single `torch.utils.data.StackDataset` and then use a dataloader.
        ...

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            returns = [evaluate_episode() for _ in range(args.evaluate_for)]

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
