#!/usr/bin/env python3
import argparse

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import npfl139
npfl139.require_version("2425.6")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--game", default="Pong", type=str, help="Game to play.")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Here you can apply wrappers to the environment if needed, such as
    # - gym.wrappers.FrameStackObservation
    # - gym.wrappers.GrayscaleObservation
    # - gym.wrappers.ResizeObservation

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

    # TODO: Train an agent using some distributed-RL algorithm.
    #
    # If you want to create N multiprocessing parallel environments, use
    #   vector_env = gym.make_vec(
    #     "ALE/{}-v5".format(args.game), N, gym.VectorizeMode.ASYNC, frameskip=args.frame_skip,
    #     vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}, wrappers=[])
    #   vector_env.reset(seed=args.seed)  # The individual environments get incremental seeds
    #
    # There are several Autoreset modes available, see https://farama.org/Vector-Autoreset-Mode.
    # In some situations, the SAME_STEP might be more practical than the default NEXT_STEP mode.
    training = True
    while training:
        ...


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("ALE/{}-v5".format(main_args.game), frameskip=main_args.frame_skip),
        main_args.seed, main_args.render_each)

    main(main_env, main_args)
