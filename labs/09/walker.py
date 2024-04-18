#!/usr/bin/env python3
import argparse
import collections
import copy

import gymnasium as gym
import numpy as np
import torch

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--envs", default=8, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=..., type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=..., type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=..., type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=..., type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=..., type=float, help="Learning rate.")
parser.add_argument("--model_path", default="walker.model", type=str, help="Model path")
parser.add_argument("--replay_buffer_size", default=1000000, type=int, help="Replay buffer size")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=..., type=float, help="Target network update weight.")


class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create an actor.
        class Actor(torch.nn.Module):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                # TODO: Create
                # - two hidden layers with `hidden_layer_size` and ReLU activation
                # - a layer for generating means with `env.action_space.shape[0]` units and no activation
                # - a layer for generating sds with `env.action_space.shape[0]` units and `torch.exp` activation
                ...

                # Then, create a variable representing a logarithm of alpha, using for example the following:
                self._log_alpha = torch.nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float32))

                # Finally, create two tensors representing the action scale and offset.
                self.register_buffer("action_scale", torch.as_tensor((env.action_space.high - env.action_space.low) / 2))
                self.register_buffer("action_offset", torch.as_tensor((env.action_space.high + env.action_space.low) / 2))

            def forward(self, inputs: torch.Tensor, sample: bool):
                # TODO: Perform the actor computation
                # - First, pass the inputs through the first hidden layer
                #   and then through the second hidden layer.
                # - From these hidden states, compute
                #   - `mus` (the means),
                #   - `sds` (the standard deviations).
                # - Then, create the action distribution using `torch.distributions.Normal`
                #   with the `mus` and `sds`.
                # - We then bijectively modify the distribution so that the actions are
                #   in the given range. Luckily, `torch.distributions.transforms` offers
                #   a class `torch.distributions.TransformedDistribution` than can transform
                #   a distribution by a given transformation. We need to use
                #   - `torch.distributions.transforms.TanhTransform()`
                #     to squash the actions to [-1, 1] range, and then
                #   - `torch.distributions.transforms.AffineTransform(self.action_offset, self.action_scale)`
                #     to scale the action ranges to [low, high].
                #   - To compose these transformations, use
                #     `torch.distributions.transforms.ComposeTransform([t1, t2], cache_size=1)`
                #     with `cache_size=1` parameter for numerical stability.
                #   Note that the `ComposeTransform` can be created already in the constructor
                #   for better performance.
                #   In case you wanted to do this manually, sample from a normal distribution, pass the samples
                #   through the `tanh` and suitable scaling, and then compute the log-prob by using `log_prob`
                #   from the normal distribution and manually accounting for the `tanh` as shown in the slides.
                #   However, the formula from the slides is not numerically stable, for a better variant see
                #   https://github.com/tensorflow/probability/blob/ef1f64a434/tensorflow_probability/python/bijectors/tanh.py#L70-L81
                # - Sample the actions by a `rsample()` call (`sample()` is not differentiable).
                # - Then, compute the log-probabilities of the sampled actions by using `log_prob()`
                #   call. An action is actually a vector, so to be precise, compute for every batch
                #   element a scalar, an average of the log-probabilities of individual action components.
                # - Finally, compute `alpha` as exponentiation of `self._log_alpha`.
                # - Return actions, log_prob, and alpha.
                #
                # Do not forget to support computation without sampling (`sample==False`). You
                # can return for example `torch.tanh(mus) * self.action_scale + self.action_offset`,
                # or you can use for example `sds=1e-7`.
                raise NotImplementedError()

        # Instantiate the actor as `self._actor`.
        self._actor = Actor(args.hidden_layer_size).apply(wrappers.torch_init_with_xavier_and_zeros).to(self.device)

        # TODO: Create a critic, which
        # - takes observations and actions as inputs,
        # - concatenates them,
        # - passes the result through two dense layers with `args.hidden_layer_size` units
        #   and ReLU activation,
        # - finally, using a last dense layer produces a single output with no activation
        # This critic needs to be cloned (for example using `copy.deepcopy`) so that
        # two critics and two target critics are created. Note that the critics should be
        # different with respect to each other, but the target critics should be the same
        # as their corresponding original critics.
        raise NotImplementedError()

        # TODO: Define an optimizer. Using `torch.optim.Adam` optimizer with
        # the given `args.learning_rate` is a good default.
        self._optimizer = ...

        # Create MSE loss.
        self._mse_loss = torch.nn.MSELoss()

    # Method for performing exponential moving average of weights of the given two modules.
    def update_parameters_by_ema(self, source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
        with torch.no_grad():
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)

    def save_actor(self, path: str):
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str):
        self._actor.load_state_dict(torch.load(path, map_location=self.device))

    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Separately train:
        # - the actor, by using two objectives:
        #   - the objective for the actor itself; in this objective, `alpha.detach()`
        #     should be used (for the `alpha` returned by the actor) to avoid optimizing `alpha`,
        #   - the objective for `alpha`, where `log_prob.detach()` should be used
        #     to avoid computing gradient for other variables than `alpha`.
        #     Use `args.target_entropy` as the target entropy (the default of -1 per action
        #     component is fine and does not need to be tuned for the agent to train).
        # - the critics using MSE loss.
        #
        # Finally, update the two target critic networks exponential moving
        # average with weight `args.target_tau`, using `self.update_parameters_by_ema`.
        raise NotImplementedError()

    # Predict actions without sampling.
    @wrappers.typed_torch_function(device, torch.float32)
    def predict_mean_actions(self, states: torch.Tensor) -> np.ndarray:
        # Return predicted actions.
        with torch.no_grad():
            return self._actor(states, sample=False)[0]

    # Predict actions with sampling.
    @wrappers.typed_torch_function(device, torch.float32)
    def predict_sampled_actions(self, states: torch.Tensor) -> np.ndarray:
        # Return sampled actions from the predicted distribution
        with torch.no_grad():
            return self._actor(states, sample=True)[0]

    @wrappers.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Produce the predicted returns, which are the minimum of
        #    target_critic(s, a) - alpha * log_prob
        #  considering both target critics and actions sampled from the actor.
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

    # Evaluation in ReCodEx
    if args.recodex:
        network.load_actor(args.model_path)
        while True:
            evaluate_episode(True)

    # Create the asynchroneous vector environment for training.
    venv = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC)

    # Replay memory of a specified maximum size.
    replay_buffer = wrappers.ReplayBuffer(max_length=args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state = venv.reset(seed=args.seed)[0]
    training, autoreset = True, np.zeros(args.envs, dtype=bool)
    while training:
        for _ in range(args.evaluate_each):
            # Predict actions by calling `network.predict_sampled_actions`.
            action = network.predict_sampled_actions(state)

            next_state, reward, terminated, truncated, _ = venv.step(action)
            done = terminated | truncated
            for i in range(args.envs):
                if not autoreset[i]:
                    replay_buffer.append(Transition(state[i], action[i], reward[i], done[i], next_state[i]))
            state = next_state
            autoreset = done

            # Training
            if len(replay_buffer) >= 10 * args.batch_size:
                # Randomly uniformly sample transitions from the replay buffer.
                batch = replay_buffer.sample(args.batch_size, np.random)
                states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
                # TODO: Perform the training

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
