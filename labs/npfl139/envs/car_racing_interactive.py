# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pygame

from . import car_racing

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_skip", default=1, type=int, help="Consider only each --frame_skip frame")
    parser.add_argument("--state", default=False, action="store_true", help="Show state instead of rendering")
    args = parser.parse_args()

    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -0.6
                if event.key == pygame.K_RIGHT:
                    a[0] = +0.6
                if event.key == pygame.K_UP:
                    a[1] = +0.5
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = car_racing.CarRacingFS(args.frame_skip, render_mode=None if args.state else "human")

    quit = False
    while not quit:
        s, info = env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            if args.state:
                env.show_state(s)
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated or restart or quit:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
