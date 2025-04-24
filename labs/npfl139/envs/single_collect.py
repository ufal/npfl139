# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import gymnasium as gym
import numpy as np

from . import multi_collect


class SingleCollect(multi_collect.MultiCollect):
    def __init__(self, render_mode: str | None = None):
        super().__init__(agents=1, render_mode=render_mode)
        self.action_space = gym.spaces.Discrete(self.action_space.nvec[0])

    def step(self, action: list[int] | np.ndarray):
        action = np.asarray(action)
        if not action.shape:
            action = np.expand_dims(action, axis=0)
        return super().step(action)


#################################
# Environment for NPFL139 class #
#################################

gym.envs.register(id="SingleCollect-v0", entry_point=SingleCollect, max_episode_steps=250, reward_threshold=0)
