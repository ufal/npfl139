# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
if __name__ == "__main__":
    import argparse
    import time

    import ale_py  # noqa: F401
    import gymnasium as gym
    import gymnasium.utils.play

    parser = argparse.ArgumentParser()
    parser.add_argument("atari_game", type=str, help="The name of the Atari game to play.")
    parser.add_argument("--frame_skip", default=1, type=int, help="The frameskip amount.")
    parser.add_argument("--zoom", default=4, type=int, help="The zoom level.")
    args = parser.parse_args()

    class ReturnReporter():
        def __init__(self):
            self.rewards = 0
            self.last_report_time = time.time()

        def __call__(self, obs_t, obs_tp1, action, rew, terminated, truncated, info):
            self.rewards += rew

            if time.time() - self.last_report_time >= 5:
                print("Partial return so far:", self.rewards)
                self.last_report_time = time.time()

            if terminated or truncated:
                print("Episode return:", self.rewards)
                self.rewards = 0
                self.last_report_time = time.time()

    env = gym.make(
        f"ALE/{args.atari_game}-v5", render_mode="rgb_array", frameskip=args.frame_skip, full_action_space=True,
    )
    if args.frame_skip > 1:
        env.unwrapped.ale.setBool("color_averaging", True)
        env.unwrapped.load_game()

    gym.utils.play.play(
        env, keys_to_action={key: int(value) for key, value in env.get_wrapper_attr("get_keys_to_action")().items()},
        callback=ReturnReporter(), fps=60 / args.frame_skip, zoom=args.zoom,
    )
