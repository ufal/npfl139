#!/usr/bin/env python3
import argparse

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import npfl139
npfl139.require_version("2526.5")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--frame_stack", default=4, type=int, help="Frame stack.")
parser.add_argument("--game", default="Pong", type=str, help="Game to play.")
parser.add_argument("--grayscale", default=True, action=argparse.BooleanOptionalAction, help="Grayscale obs.")
parser.add_argument("--screen_size", default=84, type=int, help="Screen size.")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=args.frame_skip, grayscale_obs=args.grayscale, screen_size=args.screen_size)
    env = gym.wrappers.FrameStackObservation(env, stack_size=args.frame_stack)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent

        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            while not done:
                # TODO: Choose a greedy action
                action = ...
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Train an agent using for example some distributed-RL algorithm.
    #
    # If you want to create N multithreaded parallel environments, use
    #   vector_env = ale_py.AtariVectorEnv(
    #       game=re.sub(r"(?<=[a-z])(?=[A-Z])", "_", args.game).lower(),  # use snake_case for the game name
    #       num_envs=N,  # the requred number of parallel environments,
    #       frameskip=args.frame_skip, stack_num=args.frame_stack, grayscale=args.grayscale,
    #       img_height=args.screen_size, img_width=args.screen_size,
    #       use_fire_reset=False, reward_clipping=False, repeat_action_probability=0.25,
    #       autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    #   )
    #
    # There are several Autoreset modes available, see https://farama.org/Vector-Autoreset-Mode.
    # In some situations, the SAME_STEP might be more practical than the default NEXT_STEP mode.
    training = True
    while training:
        ...


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    assert main_args.render_each in [0, 1], "Option render_each can be only 0 or 1 for Atari games"

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make(f"ALE/{main_args.game}-v5", frameskip=1, render_mode="human" if main_args.render_each else None),
        main_args.seed)

    main(main_env, main_args)
