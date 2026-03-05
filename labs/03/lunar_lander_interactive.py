#!/usr/bin/env python3
import gymnasium as gym
import gymnasium.utils.play
import pygame

class ReturnReporter():
    def __init__(self):
        self.rewards = 0

    def __call__(self, obs_t, obs_tp1, action, rew, terminated, truncated, info):
        self.rewards += rew
        if terminated or truncated:
            print("Episode reward:", self.rewards)
            self.rewards = 0

gym.utils.play.play(gym.make("LunarLander-v3", render_mode="rgb_array"),
     callback=ReturnReporter(), fps=20, zoom=2.5, noop=0, keys_to_action={
         (pygame.K_UP,): 2,
         (pygame.K_LEFT,): 1,
         (pygame.K_LEFT, pygame.K_UP): 1,
         (pygame.K_RIGHT,): 3,
         (pygame.K_RIGHT, pygame.K_UP): 3,
     })
