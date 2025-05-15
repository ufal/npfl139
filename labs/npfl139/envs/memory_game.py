# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import gymnasium as gym
import numpy as np


class MemoryGame(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, cards: int, render_mode=None):
        assert cards > 0 and cards % 2 == 0

        self._cards = cards
        self._expert = None

        self.observation_space = gym.spaces.MultiDiscrete([cards, cards // 2])
        self.action_space = gym.spaces.Discrete(cards + 1)
        self.render_mode = render_mode

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._symbols = self.np_random.permutation(np.repeat(np.arange(self._cards // 2), 2))
        self._removed = bytearray(self._cards)
        self._used = bytearray(self._cards)
        self._unused_card = 0
        self._last_card = 0

        return self.step(0)[0], {}

    def step(self, action: int):
        assert action >= 0 and action <= self._cards
        if action == 0:
            card = self._unused_card
            self._unused_card += self._unused_card + 1 < self._cards
        else:
            card = action - 1

        self._used[card] = True
        while self._unused_card + 1 < self._cards and self._used[self._unused_card]:
            self._unused_card += 1

        reward = -1
        if self._symbols[self._last_card] == self._symbols[card] \
                and self._last_card != card \
                and not self._removed[card]:
            reward = +2
            self._removed[self._last_card] = True
            self._removed[card] = True
        self._last_card = card

        if self.render_mode == "human":
            self.render()
        return np.array([card, self._symbols[card]]), reward, all(self._removed), False, {}

    def render(self, mode='human'):
        formatted = ["Memory game:"]
        for i in range(self._cards):
            formatted.append(str(self._symbols[i]) if not self._removed[i] else "X")
        formatted.append("Last card: {}".format(self._last_card))
        print(" ".join(formatted))

    def expert_episode(self) -> list[tuple[int, int]]:
        if self._expert is None:
            self._expert = make_memory_game(self._cards)
            self._expert.reset(seed=42)

        state = self._expert.reset()[0]
        episode, seen, done = [], {}, False
        while not done:
            last_card, observation = state
            if observation in seen:
                card = seen.pop(observation)
                action = 0 if card == last_card - 1 else card + 1
            else:
                seen[observation] = last_card
                action = 0

            episode.append((state, action))
            state, _, terminated, truncated, _ = self._expert.step(action)
            done = terminated or truncated
        episode.append((state, None))
        return episode


#################################
# Environment for NPFL139 class #
#################################

def make_memory_game(cards: int):
    return gym.wrappers.TimeLimit(MemoryGame(cards), max_episode_steps=2 * cards)


gym.envs.register(id="MemoryGame-v0", entry_point=make_memory_game, reward_threshold=0)
