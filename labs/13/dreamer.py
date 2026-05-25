#!/usr/bin/env python3
import argparse
import collections
import json

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
parser.add_argument("--activation", default=..., type=str, help="Non-linear activation to use.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size (number of chunks).")
parser.add_argument("--clip_gradient", default=10, type=float, help="Norm for gradient clipping.")
parser.add_argument("--chunk_length", default=16, type=int, help="Length of batch chunks.")
parser.add_argument("--evaluate_each", default=10, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--frame_skip", default=3, type=int, help="Frame skip.")
parser.add_argument("--hidden_size", default=..., type=int, help="Size of various hidden layers.")
parser.add_argument("--model_path", default="dreamer.pt", type=str, help="Path where to save the model.")
parser.add_argument("--normalization", default=..., type=str, help="Normalization of linear layers.")
parser.add_argument("--normalization_conv", default=..., type=str, help="Normalization of convolutional layers.")
parser.add_argument("--replay_buffer_size", default=100_000, type=int, help="Replay buffer capacity.")
parser.add_argument("--train_each", default=32, type=int, help="Train each given number of environment steps.")
# World-Model specific
parser.add_argument("--cnn_dim", default=..., type=int, help="Number of channels in CNN layers.")
parser.add_argument("--initial_random_episodes", default=5, type=int, help="Number of initial random episodes.")
parser.add_argument("--kl_balancing", default=0.8, type=float, help="KL balancing alpha.")
parser.add_argument("--kl_loss_weight", default=0.1, type=float, help="KL divergence loss weight.")
parser.add_argument("--latent_size", default=32, type=int, help="Size of latent state.")
parser.add_argument("--num_categories", default=32, type=int, help="Number of categorical variables.")
parser.add_argument("--wm_learning_rate", default=1e-3, type=float, help="World model learning rate.")
# Actor-Critic specific
parser.add_argument("--agent_learning_rate", default=3e-4, type=float, help="Agent learning rate.")
parser.add_argument("--entropy_penalty", default=..., type=float, help="Entropy penalty for actor loss.")
parser.add_argument("--gamma", default=..., type=float, help="Discount factor.")
parser.add_argument("--imagination_horizon", default=..., type=int, help="Horizon for imagination.")
parser.add_argument("--trace_lambda", default=..., type=float, help="Traces factor lambda.")
parser.add_argument("--target_tau", default=..., type=float, help="Target network update weight.")


def normalization(dimension: int, args: argparse.Namespace) -> torch.nn.Module:
    """Return normalization for linear layers based on the given arguments."""
    if args.normalization == "rms":
        return torch.nn.RMSNorm(dimension, eps=1e-3)
    elif args.normalization == "layer":
        return torch.nn.LayerNorm(dimension, eps=1e-3)
    elif args.normalization == "none":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported normalization: {args.normalization}")


def normalization_conv(dimension: int, args: argparse.Namespace) -> torch.nn.Module:
    """Return normalization for convolutional layers based on the given arguments."""
    if args.normalization_conv == "batch":
        return torch.nn.BatchNorm2d(dimension, eps=1e-3)
    elif args.normalization_conv == "group":
        return torch.nn.GroupNorm(max(1, dimension // 16), dimension, eps=1e-3)
    elif args.normalization_conv == "none":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported convolutional normalization: {args.normalization_conv}")


def activation(args: argparse.Namespace) -> torch.nn.Module:
    """Return non-linear activation based on the given arguments."""
    if args.activation == "relu":
        return torch.nn.ReLU()
    elif args.activation == "elu":
        return torch.nn.ELU()
    elif args.activation == "silu":
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {args.activation}")


class ObservationEncoder(torch.nn.Sequential):
    # TODO: The `ObservationEncoder` should process the given observations with
    # shape [batch_size, 3, 48, 48] in [0-1] range through a convolutional feature
    # extractor and produce an embedding of shape [batch_size, cnn_dim].
    # In the reference solution, there are three stages, each beginning with a
    # stride 2 convolution, and the first convolution has `args.cnn_dim` channels.
    # Finally, the output is flattened and passed through a linear layer to
    # produce the final embedding.
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            ...
        )


class ObservationDecoder(torch.nn.Sequential):
    # TODO: The `ObservationDecoder` should process the given state representation
    # and produce an observation with shape [batch_size, 3, 48, 48] in [0-1] range
    # (however, following the paper, I do not use any activation and use MSE loss).
    # You can start with a linear layer (followed by a normalization and activation)
    # and then reshape it (`torch.nn.Unflatten`) to a feature map with spatial dimension
    # of for example 6x6; then, follow by three stages of convolutional upsampling.
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            ...
        )


class RewardDecoder(torch.nn.Sequential):
    # TODO: The `RewardDecoder` should process the given state representation and produce a single scalar
    # reward prediction. The reference solution uses a single hidden layer and then an output layer with
    # one output and no activation.
    # BTW, if you want for the `RewardDecoder` to produce scalar rewards instead of 1-dimensional
    # reward tensors, you can use `torch.nn.Flatten(-2)` as the last layer.
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            ...
        )


class RSSM(torch.nn.Module):
    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        # TODO: Define the following RSSM components:
        # - `rnn_in` module that processes the concatenation of the previous stochastic state
        #   and the previous action before feeding it to the RNN cell. The reference solution
        #   uses a single hidden layer with a non-linear activation and `args.hidden_size` units;
        # - `cell` that is a recurrent `torch.nn.GRUCell` with `args.hidden_size` units;
        # - `prior` module representing the transition predictor, i.e., producing the
        #   prior distribution parameters from the deterministic state (but not the observation).
        #   The reference solution uses a hidden layer with a non-linear activation followed by
        #   an output linear layer with `args.latent_size * args.num_categories` units;
        # - `posterior` module representing the encoder, i.e., producing the posterior distribution
        #   parameters from the deterministic state and the observation embedding. In the reference
        #   solution, an analogous architecture to the `prior` is used.
        self.rnn_in = ...
        self.cell = ...
        self.prior = ...
        self.posterior = ...

    def sample_s(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits.unflatten(-1, (self._args.latent_size, self._args.num_categories))

        # TODO: Sample the stochastic part of the state according to `logits` with shape
        # `[batch_size, latent_size, num_categories]`, which are produced by either
        # the prior or the posterior module.
        #
        # After sampling the one-hot categorical variables (`torch.distributions.OneHotCategorical`
        # might come handy) of shape `[batch_size, latent_size, num_categories]`, apply
        # the straight-through estimator, and return the resulting sample reshaped to
        # `[batch_size, latent_size * num_categories]` together with the above `logits`.
        # (You might also try the Gumbel-softmax straight-through estimator, but it is not required.)
        sample = ...

        return sample, logits

    def step_rnn(self, h: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # TODO: Given the deterministic state `h`, the stochastic state `s`, and the
        # action `a`, compute the next deterministic state by first concatenating the
        # stochastic state and the action, processing them with `rnn_in`, and finally
        # running the GRU `cell` using `h` as the previous state and the `rnn_in` result
        # as the input.
        ...


class Actor(torch.nn.Module):
    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: Define the actor network with a shared hidden layer with a non-linear activation
        # and two heads producing the mean and standard deviation of the action distribution.
        ...

        # Finally, create two tensors representing the action scale and offset.
        self.register_buffer("action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2))
        self.register_buffer("action_offset", torch.tensor((env.action_space.high + env.action_space.low) / 2))

    def forward(self, inputs: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Process the given inputs, producing actions (sampled of the mean) and their log probabilities.
        #
        # The action distribution is a Gaussian with standard deviations processed with softplus activation,
        # followed by a tanh transformation and then an affine transformation to match the action space.
        # The reparametrization trick is required for sampling.
        #
        # Refer to the `Actor.forward` from the `walker` template which contains very detailed comments
        # on how to implement this.
        actions = ...
        log_probs = ...

        return actions, log_probs


class Critic(torch.nn.Sequential):
    # TODO: The `Critic` should process the given state and produce a single scalar value prediction.
    # The reference solution uses a single hidden layer with a non-linear activation followed by
    # an output layer. Again, you can use `torch.nn.Flatten(-2)` to produce scalar values.
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            ...
        )


class Agent(torch.nn.Module):
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        # TODO: Create the world model components and a world model optimizer.
        self.observation_encoder = ...
        self.observation_decoder = ...
        self.reward_decoder = ...
        self.rssm = ...

        self.wm_optimizer = ...

        # TODO: Create the actor, critic, target critic, and their optimizers.
        self.actor = ...
        self.critic = ...
        self.target_critic = ...

        self.actor_optimizer = ...
        self.critic_optimizer = ...

        # Optional gradient clipping
        if args.clip_gradient:
            def clip_gradients(optimizer, *_args, **_kwargs):
                torch.nn.utils.clip_grad_norm_(
                    [param for group in optimizer.param_groups for param in group["params"]], args.clip_gradient)
            for optimizer in [self.wm_optimizer, self.actor_optimizer, self.critic_optimizer]:
                optimizer.register_step_pre_hook(clip_gradients)

        # Move the agent to the device.
        self.to(self.device)

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32, torch.float32)
    def train_world_model(
        self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        self.train()

        # Name batch and time dimensions and transform observations to [0-1] range and channels-first format.
        B, T = observations.shape[:2]
        observations = observations.movedim(-1, -3) / 255

        # Given the sequences of observations, actions, and rewards with shapes [batch_size, time, ...],
        # compute the RSSM states and the world model loss.
        states_h, states_s = [], []
        loss = 0
        for t in range(T):
            # Deterministic part of the state.
            if t == 0:
                # TODO: Initialize the deterministic state `h` to zeros for the first time step.
                h = ...
            else:
                # TODO: Otherwise, compute the deterministic state `h` by calling `rssm.step_rnn`.
                h = ...

            # TODO: Compute the prior distribution parameters using `self.rssm.prior` and then
            # use `self.rssm.sample_s` to obtain prior logits (and a not-needed prior sample).
            prior_logits = ...

            # TODO: Encode the observations with `self.observation_encoder`, concatenate `h`
            # and the observation embedding, compute the posterior distribution parameters using
            # `self.rssm.posterior`, and then use `self.rssm.sample_s` to obtain the posterior
            # stochastic state sample and its logits.
            posterior_sample, posterior_logits = ...

            # We append the deterministic and stochastic states for later agent training, and
            # concatenate them to obtain the full state representation.
            states_h.append(h)
            states_s.append(posterior_sample)
            state_full = torch.cat([h, posterior_sample], dim=-1)

            # We now compute the losses. Note that for a given batch example and time step,
            # you should **sum** all the loss components (i.e., the pixels in the observation
            # and the individual distributions for the KL divergence). I achieve this by using
            # `reduction="sum"` everywhere and then dividing the final loss by `B * T` after the loop.

            # TODO: Add observation reconstruction loss using the MSE loss between the original observations
            # and the observations decoded using `self.observation_decoder` from the full state representation.
            loss += ...

            # TODO: Add reward prediction loss using the MSE loss between the original rewards and the rewards
            # decoded using `self.reward_decoder` from the full state representation.
            loss += ...

            # TODO: Given the `prior_logits` and the `posterior_logits`, add the KL divergence loss
            # weighted by `self._args.kl_loss_weight`. Use the KL balancing from the DreamerV2 paper.
            loss += ...

        # TODO: Perform a gradient step on the world model parameters using `self.wm_optimizer`
        # (so zero the gradients, call `backward` on the loss, and then step the optimizer).
        ...

        # Return the loss and the flattened deterministic and stochastic states.
        return loss, torch.stack(states_h).flatten(0, 1), torch.stack(states_s).flatten(0, 1)

    @npfl139.typed_torch_function(device, torch.float32, torch.float32)
    def train_actor_critic(self, states_h: torch.Tensor, states_s: torch.Tensor) -> tuple[float, float]:
        self.train()

        # To train the agent, we first rollout the imagination from the given states.
        imagined_states = []
        imagined_actions = []
        imagined_log_probs = []
        for _ in range(self._args.imagination_horizon):
            # TODO: Concatenate the deterministic and stochastic states to obtain full states
            # and append them to `imagined_states` for later use.
            imagined_states.append(...)

            # TODO: Sample actions from the `self.actor` by passing the full states and `sample=True`.
            # Store the sampled actions and their log probabilities in `imagined_actions` and `imagined_log_probs`.
            imagined_actions.append(...)
            imagined_log_probs.append(...)

            # TODO: Perform a step in the world model by running `self.rssm.step_rnn` and `self.rssm.sample_s`
            # to obtain the next deterministic and stochastic states.
            states_h = ...
            states_s = ...

        # TODO: Finally, create the last full state and append it to `imagined_states`
        # for bootstrapping the value of the last imagined state (however, we do not train
        # on this state, so it needs to be excluded in some of the below computations).
        imagined_states.append(...)

        # Create tensors from the collected lists.
        imagined_states = torch.stack(imagined_states)
        imagined_actions = torch.stack(imagined_actions)
        imagined_log_probs = torch.stack(imagined_log_probs)

        with torch.no_grad():
            # TODO: Compute the rewards and values for the imagined states using the
            # `self.reward_decoder` and `self.target_critic`, respectively.
            rewards = ...
            values = ...

        # TODO: Compute the lambda returns using the imagined rewards and values. The computation
        # is analogous to the `ppo` assignment, but we do not need `advantages`, just `returns`
        # (the reference solution rearranges the computation to avoid materializing the advantages).
        returns = ...

        # TODO: Train the critic. Start by computing the predicted returns using the
        # `self.critic` on the imagined states (be sure to detach the states not to
        # backpropagate though the world model and the actor), then compute the
        # MSE loss between the predicted returns and the computed returns, and finally
        # perform a gradient step using the `self.critic_optimizer`.
        ...

        # TODO: Update the target critic by performing the exponential moving average,
        # using for example the `npfl139.update_params_by_ema` utility function.
        ...

        # TODO: Train the actor. The actor loss is computed as in `ddpg`, so as the negative
        # value of `self.critic` on the imagined states (this time do not detached, the backpropagation
        # through the critic and world model is producing the desired actor gradients).
        # Don't forget to add the entropy penalty (with entropy estimated using the log probabilities
        # of the imagined actions) weighted by `self._args.entropy_penalty`. Finally, perform
        # a gradient step using the `self.actor_optimizer`.
        ...

        # Return the actor loss and the critic loss for logging purposes.
        return actor_loss, critic_loss

    def reset(self):
        # During environment interaction, we store the deterministic part of the state
        # directly in the agent. This method resets it to zeros at the beginning of each episode.
        self._state_h = torch.zeros(1, self._args.hidden_size, device=self.device)

    # Because of the decorator, the `sample` argument must always be called as a keyword (named) argument.
    @npfl139.typed_torch_function(device, torch.float32)
    def step(self, observation: torch.Tensor, sample: bool) -> tuple[np.ndarray]:
        self.eval()
        with torch.no_grad():
            # Transform the observation to [0-1] range and channels-first format, and add a batch dimension.
            observation = observation.unsqueeze(0).movedim(-1, -3) / 255

            # TODO: Now perform the step in the world model and produce the action,
            # either by sampling or by taking the mean action (according to the `sample` argument).
            #
            # Start by embedding the observation with `self.observation_encoder`.
            # Then compute the posterior distribution parameters by `self.rssm.posterior` which
            # processes the deterministic state stored in `self._state_h` and the observation embedding,
            # and sample the stochastic state using `self.rssm.sample_s`. Finally, obtain
            # the action from `self.actor` by passing the full state and the `sample` argument.
            action = ...

            # TODO: Then, perform the RSSM state update from the full state and the computed action using
            # `self.rssm.step_rnn` and store the new deterministic state in `self._state_h` for the next step.
            self._state_h = ...

            # Return the action, removing the batch dimension.
            return action.squeeze(0)

    # Serialization methods.
    def save_weights(self, path: str, acting_only: bool = False) -> None:
        """Save the model weights, either all or just the part necessary for acting."""
        if acting_only:
            children = ["actor", "rssm", "observation_encoder"]
        else:
            children = [name for name, _ in self.named_children()]
        torch.save({name: getattr(self, name).state_dict() for name in children}, path)

    def load_weights(self, path: str) -> None:
        """Load either just the actor or the full model weights."""
        for name, weights in torch.load(path, map_location=self.device).items():
            getattr(self, name).load_state_dict(weights)

    @staticmethod
    def save_args(path: str, args: argparse.Namespace) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_args(path: str) -> argparse.Namespace:
        with open(path, "r", encoding="utf-8-sig") as file:
            args = json.load(file)
        return argparse.Namespace(**args)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args if not args.recodex else Agent.load_args(args.model_path + ".json"))

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        observation = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        # TODO: Call `agent.reset()` to reset the deterministic state of the agent.
        ...

        while not done:
            # TODO: Predict an action using `agent.step` with `sample=False` argument.
            # Neither `observation` nor the returned `action` need to be transformed in any way
            # (promotion to/from batches and other conversions are handled inside `agent`).
            action = ...

            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # ReCodEx evaluation.
    if args.recodex:
        agent.load_weights(args.model_path)
        while True:
            evaluate_episode(start_evaluation=True)

    # Replay memory with the specified capacity.
    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["observation", "action", "reward", "done"])

    # Collect the initial data.
    for _ in range(args.initial_random_episodes):
        observation, done = env.reset()[0], False
        while not done:
            # TODO: Sample a random action from the environment's action space.
            action = ...

            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.append(Transition(observation, action, reward, done))

            observation = next_observation

    observation = None
    wm_loss, actor_loss, critic_loss = None, None, None
    training, steps = True, 0
    while training:
        # Step in the environment.
        if observation is None:
            # TODO: When `observation` is `None`, we need to reset both the environment and the agent.
            ...

        # TODO: Predict an action using `agent.step` with `sample=True` argument.
        # Neither `observation` nor the returned `action` need to be transformed in any way
        # (promotion to/from batches and other conversions are handled inside `agent`).
        action = ...

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.append(Transition(observation, action, reward, done))

        observation = next_observation if not done else None

        # Evaluate only when a full episode has finished (because the agent is stateful).
        if done and env.episode % args.evaluate_each == 0:
            returns = np.mean([evaluate_episode(logging=False) for _ in range(args.evaluate_for)])
            print(f"Evaluation after {env.episode} episodes: {returns:.2f};",
                  f"{wm_loss=:.4f}, {actor_loss=:.4f}, {critic_loss=:.4f}", flush=True)

        # Perform a training step when enough steps have been performed.
        steps += 1
        if steps % args.train_each == 0:
            # TODO: Sample a batch of chunks of the specified size and length from the replay buffer,
            # which now provides a method `sample_chunks(batch_size, chunk_length)`. The result
            # is a `Transition` tuple with batches of sampled items.
            ...

            # TODO: Then, train the world model using `agent.train_world_model`, storing the resulting
            # loss and also full states of the sampled chunks for agent training.
            wm_loss, states_h, states_s = agent.train_world_model(...)

            # TODO: Finally, train the actor using `agent.train_actor_critic`, storing the losses.
            actor_loss, critic_loss = agent.train_actor_critic(...)

    # Use the following code to save the hyperparameters and the model weights;
    # the `action_only` argument of `save_weights` allows saving either only
    # the part of the model necessary for acting (resulting in a smaller model)
    # or the full model (larger, but needed for `show_dreams.py`).
    #   agent.save_args(f"{path}.json", args)
    #   agent.save_weights(path, acting_only=...)

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment.
    main_env = npfl139.EvaluationEnv(
        gym.wrappers.ResizeObservation(
            gym.make("npfl139/CarRacingFS-v3", continuous=True, frame_skip=main_args.frame_skip), (48, 48)),
        main_args.seed, main_args.render_each)

    main(main_env, main_args)
