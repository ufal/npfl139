# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import gymnasium as gym
import numpy as np
import pygame

if __name__ == "__main__":
    env = gym.make("SingleCollect-v0", render_mode="human")
    env.reset()

    quit = False
    while not quit:
        env.reset()
        steps, action, restart, rewards = 0, 0, False, []
        while True:
            # Handle input
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 2
                    if event.key == pygame.K_RIGHT:
                        action = 1
                    if event.key == pygame.K_UP:
                        action = 3
                    if event.key == pygame.K_DOWN:
                        action = 4
                    if event.key == pygame.K_RETURN:
                        restart = True
                    if event.key == pygame.K_ESCAPE:
                        quit = True
                if event.type == pygame.QUIT:
                    quit = True

            # Perform the step
            _, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated or restart or quit:
                break

            steps += 1
            action = 0
            rewards.append(reward)
            if len(rewards) % 25 == 0:
                print("Rewards for last 25 timesteps: {}".format(np.sum(rewards[-25:])))
        print("Episode ended with a return of {}".format(np.sum(rewards)))

    env.close()
