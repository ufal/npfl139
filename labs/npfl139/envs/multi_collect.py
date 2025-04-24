# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import gymnasium as gym
import numpy as np


class MultiCollect(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, agents: int, render_mode: str | None = None):
        assert agents > 0
        self._agents = agents

        self.observation_space = gym.spaces.Box(
            np.array([*[-10, -10] * agents, *[-np.inf, -np.inf] * agents, *[-0.5, -0.5] * agents], np.float32),
            np.array([*[+10, +10] * agents, *[+np.inf, +np.inf] * agents, *[+0.5, +0.5] * agents], np.float32))
        self.action_space = gym.spaces.MultiDiscrete([5] * agents)
        self.render_mode = render_mode
        self._screen = None
        self._surface = None

    def reset(self, seed=None, options=None):
        self._agents_pos = self.np_random.uniform(-10, 10, size=[self._agents, 2]).astype(np.float32)
        self._agents_vel = self.np_random.integers(-4, 4 + 1, size=[self._agents, 2])
        self._centers = self.np_random.uniform(-10, 10, size=[self._agents, 2]).astype(np.float32)
        self._centers_hit = np.zeros(self._agents, dtype=np.int32)

        observation, _, _, _, info = self.step(np.zeros(self._agents, dtype=np.int32))

        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: list[int] | np.ndarray):
        action = np.asarray(action)
        assert len(action) == self._agents
        assert np.all(action >= 0) and np.all(action < 5)

        # Update speeds
        self._agents_vel[:, 0] += (action == 1)
        self._agents_vel[:, 0] -= (action == 2)
        self._agents_vel[:, 1] += (action == 3)
        self._agents_vel[:, 1] -= (action == 4)
        self._agents_vel = np.clip(self._agents_vel, -4, 4)

        # Update pos
        self._agents_pos += self._agents_vel / 8

        # Compute reward and update the hit information
        rewards = np.zeros(self._agents)
        distances = np.linalg.norm(self._centers[:, np.newaxis] - self._agents_pos[np.newaxis, :], axis=-1)
        for i in range(self._agents):
            a = np.argmin(distances[i])
            closest = distances[i][a]
            if closest < 1:
                rewards[a] += 1
                self._centers_hit[i] += 1
                if self._centers_hit[i] >= 10:
                    rewards[a] += 50
                    self._centers_hit[i] = 0
                    self._centers[i] = self.np_random.uniform(-10, 10, size=2)
            else:
                rewards[a] += 1 - (closest - 1) / 10
                self._centers_hit[i] = 0

        distances = np.linalg.norm(self._agents_pos[:, np.newaxis] - self._agents_pos[np.newaxis, :], axis=-1)
        closest = np.min(distances + np.eye(self._agents), axis=1)
        self._agents_hit = closest < 1
        rewards -= self._agents_hit
        observation = np.concatenate(
            [self._centers.ravel(), self._agents_pos.ravel(), self._agents_vel.ravel().astype(np.float32) / 8])

        if self.render_mode == "human":
            self.render()
        return observation, np.mean(rewards), False, False, {"agent_rewards": rewards}

    def render(self):
        return self._render(self.render_mode)

    def _render(self, mode: str):
        import pygame
        import pygame.gfxdraw

        assert mode in self.metadata["render_modes"]

        W, R = 600, 30
        if self._surface is None:
            self._surface = pygame.Surface((W, W))
            if mode == "human":
                pygame.init()
                self._screen = pygame.display.set_mode((W, W))
                self._clock = pygame.time.Clock()

        self._surface.fill((255, 255, 255))
        for (x, y), hit in zip(self._centers, self._centers_hit):
            for method in [pygame.gfxdraw.aacircle, pygame.gfxdraw.filled_circle]:
                method(self._surface, W // 2 + int(x * R), W // 2 - int(y * R), R, (0, 255 if hit else 180, 0))
        for (x, y), hit in zip(self._agents_pos, self._agents_hit):
            for method in [pygame.gfxdraw.aacircle, pygame.gfxdraw.filled_circle]:
                method(self._surface, W // 2 + int(x * R), W // 2 - int(y * R), R // 2,
                       (255, 0, 0) if hit else (0, 0, 180))
        for (x, y), (xv, yv) in zip(self._agents_pos, self._agents_vel):
            pygame.draw.line(self._surface, (0, 0, 0), (W // 2 + int(x * R), W // 2 - int(y * R)),
                             (W // 2 + int((x + xv) * R), W // 2 - int((y + yv) * R)), width=2)

        if mode == "human":
            pygame.event.pump()
            self._clock.tick(self.metadata["render_fps"])
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._surface)), axes=(1, 0, 2)
            )

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


#################################
# Environment for NPFL139 class #
#################################

gym.envs.register(id="MultiCollect-v0", entry_point=MultiCollect, max_episode_steps=250, reward_threshold=0)
