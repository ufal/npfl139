#!/usr/bin/env python3
import argparse
import collections
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.6")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the loss computation")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--epsilon", default=..., type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=None, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=..., type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=..., type=int, help="Size of hidden layer.")
parser.add_argument("--kappa", default=..., type=float, help="The quantile Huber loss threshold.")
parser.add_argument("--learning_rate", default=..., type=float, help="Learning rate.")
parser.add_argument("--quantiles", default=..., type=int, help="Number of quantiles.")
parser.add_argument("--target_update_freq", default=..., type=int, help="Target update frequency.")


class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model and store it as `self._model`. The model
        # should compute `args.quantiles` quantiles for each action, so for input
        # of shape `[batch_size, *env.observation_space.shape]`, the output should
        # have the shape `[batch_size, env.action_space.n, args.quantiles]`.
        # The module `torch.nn.Unflatten` might come handy.
        self._model = torch.nn.Sequential(
            ...
        )
        self._model.to(self.device)

        # Store the discount factor and the quantile Huber loss threshold.
        self.gamma = args.gamma
        self.kappa = args.kappa

        # TODO(q_network): Define a suitable optimizer from `torch.optim`.
        self._optimizer = ...

    @staticmethod
    def compute_loss(
        states_quantiles: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
        next_states_quantiles: torch.Tensor, gamma: float, kappa: float,
    ) -> torch.Tensor:
        # TODO: Implement the loss computation according to the QR-DQN-kappa algorithm.
        # - The `states_quantiles` are current state quantiles, of shape `[batch_size, actions, quantiles]`.
        # - The `actions` are the integral actions taken in the states, of shape `[batch_size]`.
        # - The `rewards` are the rewards obtained after taking the actions, of shape `[batch_size]`.
        # - The `dones` are `torch.float32` indicating whether the episode ended, of shape `[batch_size]`.
        # - The `next_states_quantiles` are next states quantiles, of shape `[batch_size, actions, quantiles]`.
        #   Because they should not be backpropagated through, use an appropriate `.detach()` call.
        # - The non-negative `kappa` is the threshold for the quantile Huber loss (delta in PyTorch terminology).
        #   When `kappa=0` is passed, the standard (non-Huber) quantile regression loss should be used.
        # The number of quantiles is given by the shape of `states_quantiles`, and the quantiles
        # tau_1, ..., tau_N are uniformly spaced between 0 (exclusive) and 1 (inclusive), so tau_i = i / N.
        # The resulting loss should be the mean over all trained quantiles and all batch examples,
        # unlike the algorithm in the paper, which computes a sum over the trained quantiles.
        return ...

    # The training function defers the computation to the `compute_loss` method.
    #
    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
              dones: torch.Tensor, next_states: torch.Tensor) -> None:
        self._model.train()
        # Pass all arguments to the `compute_loss` method.
        loss = self.compute_loss(
            self._model(states), actions, rewards, dones, self._model(next_states), self.gamma, self.kappa)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            # TODO: Return all predicted Q-values for the given states.
            return ...

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> Callable | None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # When the `args.verify` is set, just return the loss computation function for validation.
    if args.verify:
        return Network.compute_loss

    # Construct the network
    network = Network(env, args)

    # Replay memory; the `max_length` parameter can be passed to limit its size.
    replay_buffer = npfl139.ReplayBuffer()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon
    training = True
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO(q_network): Choose an action.
            # You can compute the q_values of a given state by
            #   q_values = network.predict(state[np.newaxis])[0]
            action = ...

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the `replay_buffer` is large enough, perform training by
            # sampling a batch of `args.batch_size` uniformly randomly chosen transitions
            # and calling `network.train(states, actions, rewards, dones, next_states)`.

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(q_network): Choose (greedy) action
            action = ...
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    result = main(main_env, main_args)
    if main_args.verify:
        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-1.4, 0.1, 0.8], [-1.2, 0.1, 1.1]]]),
            actions=torch.tensor([1]), rewards=torch.tensor([-1.5]), dones=torch.tensor([0.]),
            next_states_quantiles=torch.tensor([[[-0.4, 0.1, 0.4], [-0.5, 1.0, 1.6]]]),
            gamma=0.2, kappa=1.5).numpy(force=True), 0.3294963, atol=1e-5)

        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-0.0, 0.1, 1.2], [-1.8, -0.2, -0.1]],
                                           [[-0.3, 0.5, 1.3], [-1.4, -0.7, -0.1]],
                                           [[-0.3, -0.0, 1.9], [-1.1, -0.2, -0.1]]]),
            actions=torch.tensor([1, 0, 1]), rewards=torch.tensor([0.5, 1.4, 0.1]), dones=torch.tensor([0., 0., 1.]),
            next_states_quantiles=torch.tensor([[[-1.1, 0.2, 0.3], [-0.4, 1.1, 1.3]],
                                                [[-0.6, -0.5, 2.0], [-0.3, 0.2, 0.4]],
                                                [[-0.9, 0.7, 2.3], [-0.3, 0.7, 0.7]]]),
            gamma=0.8, kappa=0.0).numpy(force=True), 0.4392593, atol=1e-5)

        np.testing.assert_allclose(result(
            states_quantiles=torch.tensor([[[-0.8, -0.5, -0.0, 0.3], [-0.7, -0.2, -0.2, 1.6]],
                                           [[-1.5, -1.4, -0.6, 0.1], [-2.1, -1.5, -0.3, 0.3]]]),
            actions=torch.tensor([1, 0]), rewards=torch.tensor([-0.0, 0.7]), dones=torch.tensor([1., 0.]),
            next_states_quantiles=torch.tensor([[[-1.2, 0.3, 0.4, 0.7], [-1.2, -0.1, 0.4, 2.2]],
                                                [[-1.5, 0.2, 0.2, 0.5], [-0.9, 0.4, 0.5, 1.3]]]),
            gamma=0.3, kappa=3.5).numpy(force=True), 0.2906375, atol=1e-5)
