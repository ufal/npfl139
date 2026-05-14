#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl139

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--activation", default=..., type=str, help="Normalization to use.")
parser.add_argument("--batch_size", default=..., type=int, help="Batch size (number of chunks).")
parser.add_argument("--chunk_length", default=..., type=int, help="Length of batch sequences.")
parser.add_argument("--frame_skip", default=..., type=int, help="Frame skip.")
parser.add_argument("--hidden_size", default=..., type=int, help="Hidden layer size.")
parser.add_argument("--learning_rate", default=..., type=float, help="Learning rate.")
parser.add_argument("--normalization", default=..., type=str, help="Normalization to use.")
parser.add_argument("--replay_buffer_size", default=..., type=int, help="Replay buffer capacity.")
# RSSM specific
parser.add_argument("--latent_size", default=..., type=int, help="Size of latent state.")
parser.add_argument("--num_categories", default=..., type=int, help="Number of categorical variables.")
parser.add_argument("--free_nats", default=..., type=float, help="Free nats/bits clipping for KL.")
# Dreamer Actor-Critic specific
parser.add_argument("--imagination_horizon", default=..., type=int, help="Horizon H for imagination.")
parser.add_argument("--gamma", default=..., type=float, help="Discount factor.")
parser.add_argument("--lambda_", default=..., type=float, help="Lambda for TD(lambda) returns.")
parser.add_argument("--actor_lr", default=..., type=float, help="Actor learning rate.")
parser.add_argument("--critic_lr", default=..., type=float, help="Critic learning rate.")


class Encoder(torch.nn.Sequential):
    def __init__(self, args):
        ...


class Decoder(torch.nn.Sequential):
    def __init__(self, args):
        ...


class RSSM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

    def sample_z(self, x):
        ...

    def step_rnn(self, s, a, h):
        ...


class Actor(torch.nn.Module):
    def __init__(self, args):
        ...

    def forward(self, inputs):
        ...


class Critic(torch.nn.Module):
    def __init__(self, args):
        ...

    def forward(self, inputs):
        ...


class Agent:
    def __init__(self, env, args):
        ...

    def reset_state(self):
        ...

    def train_world_model(self, inputs):
        ...

    def train_actor_critic(self, inputs):
        # 1. Rollout Imagination
        ...

        # 2. Compute Rewards and Values
        ...

        # 3. Compute Lambda Returns
        ...

        # 4. Train Critic
        ...

        # 5. Train Actor
        ...

    def act(self, inputs):
        ...


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Collect initial data

    training = True
    while training:
        # Perform some number of steps in the environment
        ...

        # Train the RSSM on the collected data
        ...

        # Train the Actor and Critic using imagined trajectories
        ...

        # Once in a while, evaluate the current policy
        ...


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("npfl139/CarRacingFS-v3", continuous=True, frame_skip=main_args.frame_skip),
        main_args.seed, main_args.render_each)

    main(main_env, main_args)
