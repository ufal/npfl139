# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import gymnasium as gym
import numpy as np


class DiscretizationWrapper(gym.ObservationWrapper):
    def __init__(self, env, separators, tiles=None):
        super().__init__(env)
        self._separators = separators
        self._tiles = tiles

        if tiles is None:
            states = 1
            for separator in separators:
                states *= 1 + len(separator)
            self.observation_space = gym.spaces.Discrete(states)
        else:
            self._first_tile_states, self._rest_tiles_states = 1, 1
            for separator in separators:
                self._first_tile_states *= 1 + len(separator)
                self._rest_tiles_states *= 2 + len(separator)
            self.observation_space = gym.spaces.MultiDiscrete([
                self._first_tile_states + i * self._rest_tiles_states for i in range(tiles)])

            self._separator_offsets = [0 if len(s) <= 1 else (s[1] - s[0]) / tiles for s in separators]
            self._separator_tops = [np.inf if len(s) <= 1 else s[-1] + (s[1] - s[0]) for s in separators]

    def observation(self, observations):
        state = 0
        for observation, separator in zip(observations, self._separators):
            state *= 1 + len(separator)
            state += np.digitize(observation, separator)
        if self._tiles is None:
            return state
        else:
            states = np.empty(self._tiles, dtype=np.int64)
            states[0] = state
            for t in range(1, self._tiles):
                state = 0
                for i in range(len(self._separators)):
                    state *= 2 + len(self._separators[i])
                    value = observations[i] + ((t * (2 * i + 1)) % self._tiles) * self._separator_offsets[i]
                    if value > self._separator_tops[i]:
                        state += 1 + len(self._separators[i])
                    else:
                        state += np.digitize(value, self._separators[i])
                states[t] = self._first_tile_states + (t - 1) * self._rest_tiles_states + state
            return states


class DiscreteCartPoleWrapper(DiscretizationWrapper):
    def __init__(self, env, bins=8):
        super().__init__(env, [
            np.linspace(-2.4, 2.4, num=bins + 1)[1:-1],  # cart position
            np.linspace(-3, 3, num=bins + 1)[1:-1],      # cart velocity
            np.linspace(-0.2, 0.2, num=bins + 1)[1:-1],  # pole angle
            np.linspace(-2, 2, num=bins + 1)[1:-1],      # pole angle velocity
        ])


class DiscreteMountainCarWrapper(DiscretizationWrapper):
    def __init__(self, env, bins=None, tiles=None):
        if bins is None:
            bins = 24 if tiles is None or tiles <= 1 else 12 if tiles <= 3 else 8
        super().__init__(env, [
            np.linspace(-1.2, 0.6, num=bins + 1)[1:-1],    # car position
            np.linspace(-0.07, 0.07, num=bins + 1)[1:-1],  # car velocity
        ], tiles)


class DiscreteLunarLanderWrapper(DiscretizationWrapper):
    def __init__(self, env):
        super().__init__(env, [
            np.linspace(-.4, .4, num=5 + 1)[1:-1],      # position x
            np.linspace(-.075, 1.35, num=6 + 1)[1:-1],  # position y
            np.linspace(-.5, .5, num=5 + 1)[1:-1],      # velocity x
            np.linspace(-.8, .8, num=7 + 1)[1:-1],      # velocity y
            np.linspace(-.2, .2, num=3 + 1)[1:-1],      # rotation
            np.linspace(-.2, .2, num=5 + 1)[1:-1],      # ang velocity
            [.5],                                       # left contact
            [.5],                                       # right contact
        ])

        self._expert = gym.make("LunarLander-v3")
        gym.Env.reset(self._expert.unwrapped, seed=42)

    def expert_trajectory(self, seed=None):
        state, trajectory, done = self._expert.reset(seed=seed)[0], [], False
        while not done:
            action = gym.envs.box2d.lunar_lander.heuristic(self._expert, state)
            next_state, reward, terminated, truncated, _ = self._expert.step(action)
            trajectory.append((self.observation(state), action, reward))
            done = terminated or truncated
            state = next_state
        trajectory.append((self.observation(state), None, None))
        return trajectory


gym.envs.register(
    id="MountainCar1000-v0",
    entry_point="gymnasium.envs.classic_control.mountain_car:MountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)
