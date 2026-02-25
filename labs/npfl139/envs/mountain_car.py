# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import gymnasium as gym


# NPFL139 environment registration.
gym.envs.register(
    id="npfl139/MountainCar1000-v0",
    entry_point="gymnasium.envs.classic_control.mountain_car:MountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)
