#!/usr/bin/env python3
import argparse
import os

import gymnasium as gym
import numpy as np
import torch

import memory_game_environment
import wrappers
memory_game_environment.register()

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=4, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=..., type=int, help="Number of episodes to train on.")
parser.add_argument("--gradient_clipping", default=1.0, type=float, help="Gradient clipping.")
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=..., type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=..., type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=..., type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=..., type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=..., type=int, help="Memory cell size; default 3/2*`cards`")
parser.add_argument("--replay_buffer", default=..., type=int, help="Max replay buffer size; default batch_size")

class Network:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env

        class Agent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # TODO: Create suitable layers.
                raise NotImplementedError()

            def forward(self, memory, state):
                # Encode the input state, which is a (card, observation) pair,
                # by representing each element as one-hot and concatenating them, resulting
                # in a vector of length `sum(env.observation_space.nvec)`.
                encoded_input = torch.cat([torch.nn.functional.one_hot(torch.relu(state[:, i]), dim).float()
                                           for i, dim in enumerate(env.observation_space.nvec)], dim=-1)

                # TODO(memory game): Generate a read key for memory read from the encoded input, by using
                # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
                # with `args.memory_cell_size` units and `tanh` activation (to keep the memory
                # content in limited range).

                # TODO(memory game): Read the memory using the generated read key. Notably, compute cosine
                # similarity of the key and every memory row, apply softmax to generate
                # a weight distribution over the rows, and finally take a weighted average of
                # the memory rows.

                # TODO(memory game): Using concatenated encoded input and the read value, use a ReLU hidden
                # layer of size `args.hidden_layer` followed by a dense layer with
                # `env.action_space.n` units to produce policy logits.

                # TODO(memory game): Perform memory write. For faster convergence, append directly
                # the `encoded_input` to the memory, i.e., add it as a first memory row, and drop
                # the last memory row to keep memory size constant.

                # TODO(memory game): Return the updated memory and the policy
                raise NotImplementedError()

        # Create the agent
        self._agent = Agent()
        self._agent.apply(wrappers.torch_init_with_xavier_and_zeros).to(self.device)

        # TODO: Create an optimizer and a loss function
        raise NotImplementedError()

    def zero_memory(self):
        # TODO(memory_game): Return an empty memory. It should be a TF tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]` on `self.device`.
        raise NotImplementedError()

    @wrappers.typed_torch_function(device, torch.int64, torch.int64, torch.float32)
    def _train(self, states, actions, returns):
        # TODO: Train the network given a batch of sequences of `states`
        # (each being a (card, symbol) pair), sampled `actions` and observed `returns`.
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `actions` as actions.
        #
        # Use the REINFORCE algorithm, optionally with a baseline. Note that
        # I use a baseline, but not a baseline computed by a neural network;
        # instead, for every time step, I track exponential moving average of
        # observed returns, with momentum 0.01. Furthermore, I use entropy regularization
        # with coefficient `args.entropy_regularization`.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).
        raise NotImplementedError()

    def train(self, episodes):
        # TODO: Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        raise NotImplementedError()

    @wrappers.typed_torch_function(device, torch.float32, torch.int64)
    def predict(self, memory, state):
        self._agent.eval()
        with torch.no_grad():
            memory, logits = self._agent(memory, state)
            return memory, torch.softmax(logits, dim=-1)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Post-process arguments to default values if not overridden on the command line.
    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    if args.replay_buffer is None:
        args.replay_buffer = args.batch_size
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], network.zero_memory()
        rewards, done = 0, False
        while not done:
            # TODO(memory_game): Find out which action to use
            action = ...
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Training
    replay_buffer = wrappers.ReplayBuffer(max_length=args.replay_buffer)
    training = True
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each):
            state, memory, episode, done = env.reset()[0], network.zero_memory(), [], False
            while not done:
                # TODO: Choose an action according to the generated distribution.
                action = None

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode.append([state, action, reward])
                state = next_state

            # TODO: In the `episode`, compute returns from the rewards.

            replay_buffer.append(episode)

            # Train the network if enough data is available
            if len(replay_buffer) >= args.batch_size:
                network.train(replay_buffer.sample(args.batch_size, np.random, replace=False))

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("MemoryGame-v0", cards=args.cards), args.seed, args.render_each,
                                 evaluate_for=args.evaluate_for, report_each=args.evaluate_for)

    main(env, args)
