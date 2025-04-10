#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2425.8")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
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
parser.add_argument("--tiles", default=..., type=int, help="Tiles to use.")


class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Analogously to paac, your model should contain two components:
        # - an actor, which predicts distribution over the actions, and
        # - a critic, which predicts the value function.
        #
        # The given states are tile encoded, so they are integer indices of
        # tiles intersecting the state. Therefore, you should convert them
        # to dense encoding (one-hot-like, with `args.tiles` ones); or you can
        # even use the `torch.nn.EmbeddingBag` layer.
        #
        # The actor computes `mus` and `sds`, each of shape `[batch_size, actions]`.
        # Compute each independently using states as input, adding a fully connected
        # layer with `args.hidden_layer_size` units and a ReLU activation. Then:
        # - For `mus`, add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required range, you should apply
        #   properly scaled `torch.tanh` activation.
        # - For `sds`, add a fully connected layer with `actions` outputs
        #   and `torch.exp` or `torch.nn.functional.softplus` activation.
        #
        # The critic should be a usual one, passing states through one hidden
        # layer with `args.hidden_layer_size` ReLU units and then predicting
        # the value function.
        raise NotImplementedError()

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.int64, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Run the model on given `states` and compute `sds`, `mus` and predicted values.
        # Then create `action_distribution` using `torch.distributions.Normal` class and
        # the computed `mus` and `sds`.
        #
        # TODO: Train the actor using the sum of the following two losses:
        # - REINFORCE loss, i.e., the negative log likelihood of the `actions` in the
        #   `action_distribution` (using the `log_prob` method). You then need to sum
        #   the log probabilities of the action components in a single batch example.
        #   Finally, multiply the resulting vector by `(returns - baseline)`
        #   and compute its mean. Be sure to let the gradient flow only where it should.
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        #
        # Train the critic using mean square error of the `returns` and predicted values.
        raise NotImplementedError()

    @npfl139.typed_torch_function(device, torch.int64)
    def predict_actions(self, states: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return predicted action distributions (mus and sds).
        raise NotImplementedError()

    @npfl139.typed_torch_function(device, torch.int64)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted state-action values.
        raise NotImplementedError()


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
            # TODO: Predict an action using the greedy policy.
            action = ...
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment, using the SAME_STEP autoreset mode.
    vector_env = gym.make_vec("MountainCarContinuous-v0", args.envs, gym.VectorizeMode.ASYNC,
                              wrappers=[lambda env: npfl139.DiscreteMountainCarWrapper(env, tiles=args.tiles)],
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    states = vector_env.reset(seed=args.seed)[0]

    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Predict action distribution using `agent.predict_actions`
            # and then sample it using for example `np.random.normal`. Do not
            # forget to clip the actions to the `env.action_space.{low,high}`
            # range, for example using `np.clip`.
            actions = ...

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated

            # TODO(paac): Compute estimates of returns by one-step bootstrapping

            # TODO(paac): Train agent using current states, chosen actions and estimated returns.
            ...

            states = next_states

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=main_args.tiles),
        main_args.seed, main_args.render_each)

    main(main_env, main_args)
