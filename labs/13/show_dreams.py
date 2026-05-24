#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import pygame
import torch

import npfl139
npfl139.require_version("2526.14.0")

import dreamer

parser = argparse.ArgumentParser()
parser.add_argument("--frame_skip", default=None, type=int, help="Frame skip.")
parser.add_argument("--model_path", default="dreamer_full.pt", type=str, help="Path to the saved full model.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


BASE_SIZE = 48
SCALE = 8
RENDER_SIZE = BASE_SIZE * SCALE


def action_to_tensor(action, device):
    return torch.from_numpy(action).float().unsqueeze(0).to(device)


def obs_to_tensor(obs, device):
    tensor = torch.from_numpy(obs).float().unsqueeze(0).movedim(-1, -3) / 255.0
    return tensor.to(device)


def obs_to_surface(obs):
    surface = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
    surface = pygame.transform.scale(surface, (RENDER_SIZE, RENDER_SIZE))
    return surface


def tensor_to_surface(tensor):
    img = tensor.squeeze(0).movedim(-3, -1).numpy(force=True)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return obs_to_surface(img)


def main(cmd_args: argparse.Namespace) -> None:
    # Load agent options (the model frame skip is needed to create the environment).
    args = dreamer.Agent.load_args(cmd_args.model_path + ".json")
    if cmd_args.frame_skip is not None:  # override frame skip if provided via command line
        args.frame_skip = cmd_args.frame_skip

    # Create the environment.
    env = gym.make("npfl139/CarRacingFS-v3", continuous=True, frame_skip=args.frame_skip)
    env = gym.wrappers.ResizeObservation(env, (BASE_SIZE, BASE_SIZE))

    # Load the agent, which must contain the full model.
    agent = dreamer.Agent(env, args)
    agent.load_weights(cmd_args.model_path)
    agent.eval()

    # Initialize the visualization.
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 36)
    screen = pygame.display.set_mode((RENDER_SIZE * 3, RENDER_SIZE))
    pygame.display.set_caption("Dreamer Interactive Visualization")

    FPS = 30 // (args.frame_skip or 1)
    clock = pygame.time.Clock()

    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    imagination_step_count = 0
    h_prior, s_prior = None, None
    restart = True
    stop = False
    while not stop:
        # Restart if needed.
        if restart:
            obs, _ = env.reset()
            h_current = torch.zeros(1, args.hidden_size, device=agent.device)
            with torch.no_grad():
                obs_tensor = obs_to_tensor(obs, agent.device)
                obs_embed = agent.observation_encoder(obs_tensor)
                s_current, _ = agent.rssm.get_s(agent.rssm.posterior(torch.cat([h_current, obs_embed], dim=-1)))
            imagination_step_count = 0
            h_prior, s_prior = h_current.clone(), s_current.clone()
            restart = False

        # Perform one step in the environment.
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            restart = True
            continue

        action_tensor = action_to_tensor(action, agent.device)
        obs_tensor = obs_to_tensor(obs, agent.device)

        with torch.no_grad():
            # Update the posterior (state based on the actual observation).
            h_current = agent.rssm.step_rnn(h_current, s_current, action_tensor)
            obs_embed = agent.observation_encoder(obs_tensor)
            s_current, _ = agent.rssm.get_s(agent.rssm.posterior(torch.cat([h_current, obs_embed], dim=-1)))

            # Update the prior (pure imagination).
            h_prior = agent.rssm.step_rnn(h_prior, s_prior, action_tensor)
            s_prior, _ = agent.rssm.get_s(agent.rssm.prior(h_prior))

            # Render all the observations.
            screen.fill((0, 0, 0))

            surface_left = obs_to_surface(obs)
            screen.blit(surface_left, (0, 0))

            decoded_post = agent.observation_decoder(torch.cat([h_current, s_current], dim=-1))
            surface_mid = tensor_to_surface(decoded_post)
            screen.blit(surface_mid, (RENDER_SIZE, 0))

            decoded_prior = agent.observation_decoder(torch.cat([h_prior, s_prior], dim=-1))
            surface_right = tensor_to_surface(decoded_prior)
            screen.blit(surface_right, (RENDER_SIZE * 2, 0))

            imagination_step_count += 1
            text_surface = font.render(f"Imagination Steps: {imagination_step_count}\nFPS: {FPS}", True, (255, 255, 0))
            screen.blit(text_surface, (10, 10))

            # Limit the frame rate.
            clock.tick(FPS)

        pygame.display.flip()

        # Process the input events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = -0.8
                if event.key == pygame.K_RIGHT:
                    action[0] = +0.8
                if event.key == pygame.K_UP:
                    action[1] = +0.8
                if event.key == pygame.K_DOWN:
                    action[2] = +0.8
                if event.key == pygame.K_MINUS:
                    FPS = max(1, FPS - 1)
                if event.key in [pygame.K_EQUALS, pygame.K_PLUS]:
                    FPS += 1
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    stop = True
                if event.key == pygame.K_SPACE:  # restart imagination
                    imagination_step_count = 0
                    h_prior, s_prior = h_current.clone(), s_current.clone()

            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                    action[0] = 0
                if event.key == pygame.K_UP:
                    action[1] = 0
                if event.key == pygame.K_DOWN:
                    action[2] = 0


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    main(main_args)
