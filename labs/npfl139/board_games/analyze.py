#!/usr/bin/env python3
#
# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import argparse
import dataclasses
import importlib
import os
import time

import numpy as np
import pygame

from .board_game import BoardGame


def load_player(args: argparse.Namespace, player: str):
    player, *player_args = player.split(":")

    # Load a player from a module
    if player.endswith(".py"):
        player = player[:-3]

    module = importlib.import_module(player)
    module_args = module.parser.parse_args(player_args)
    module_args.recodex = True
    if hasattr(module_args, "seed") and module_args.seed is None and args.seed is not None:
        module_args.seed = args.seed
    try:
        cwd = os.getcwd()
        os.chdir(os.path.dirname(module.__file__))
        return module.main(module_args)
    finally:
        os.chdir(cwd)

    raise ValueError(f"Could not load player {player}")


@dataclasses.dataclass
class HistoryEntry:
    game: BoardGame
    parent: int
    policy: np.ndarray
    value: float
    values: np.ndarray | None = None
    mcts: np.ndarray | None = None


class Analyzer:
    def __init__(self, game_start, player, mcts_simulations):
        self.game_start = game_start
        self.player = player
        self.mcts_simulations = mcts_simulations

        self.history = [HistoryEntry(self.game_start, -1, *self.player.evaluate(self.game_start))]
        self.new_game()

        self.players, self.player_types = [0, 0], ["user", "auto"]
        self.value_type, self.value_types = 1, ["none", "policy", "values", "mcts"]
        self.buttons = [
            (lambda: self.player_types[self.players[0]].title()[:4],
             lambda _: setattr(self, "players", [self.players[0] ^ 1, self.players[1]])),
            (lambda: self.player_types[self.players[1]].title()[:4],
             lambda _: setattr(self, "players", [self.players[0], self.players[1] ^ 1])),
            (lambda: self.value_types[self.value_type].title()[:4],
             lambda b: setattr(self, "value_type", (self.value_type + (1 if b == 1 else -1)) % 4)),
            (lambda: f"{100 * self.history[self.current].value:04.1f}", lambda _: None),
            (lambda: "<<", lambda _: setattr(self, "current", self.change_current(lambda c: c - 1))),
            (lambda: "^", lambda _: setattr(self, "current", self.change_current(lambda c: self.history[c].parent))),
            (lambda: ">>", lambda _: setattr(self, "current", self.change_current(lambda c: c + 1))),
            (lambda: "New", lambda _: self.new_game()),
        ]

        pygame.init()
        self.game_size = game_start.render_size()
        self.button_size = (self.game_size[0] // len(self.buttons), self.game_size[0] // 8)
        self.screen = pygame.display.set_mode((self.game_size[0], self.game_size[1] + self.button_size[1]))
        self.surface = pygame.Surface(self.screen.size)

        pygame.freetype.init()
        self.font = pygame.freetype.SysFont(None, self.button_size[1] * 2 // 5)

    def new_game(self):
        self.current = 0

    def move(self, action):
        game = self.history[self.current].game.clone()
        game.move(action)
        for i in range(len(self.history)):
            if np.array_equal(game.board, self.history[i].game.board) and self.current == self.history[i].parent:
                self.current = i
                break
        else:
            self.history.append(HistoryEntry(game, self.current, *self.player.evaluate(game) if not game.outcome() else (None, 0)))
            self.current = len(self.history) - 1
        self.render()

    def change_current(self, new_current_fn):
        new_current = new_current_fn(self.current)
        if 0 <= new_current < len(self.history) and self.players[self.history[new_current].game.to_play]:
            new_current = new_current_fn(new_current)
        return max(0, min(len(self.history) - 1, new_current))

    def render(self):
        if not self.history[self.current].game.outcome():
            if self.value_types[self.value_type] == "values" and self.history[self.current].values is None:
                self.history[self.current].values = self.player.evaluate_values(self.history[self.current].game)
            if self.value_types[self.value_type] == "mcts" and self.history[self.current].mcts is None:
                self.history[self.current].mcts = self.player.evaluate_mcts(self.history[self.current].game, self.mcts_simulations)

        self.surface.fill((0, 0, 0))
        values = getattr(self.history[self.current], self.value_types[self.value_type], None)
        self.history[self.current].game.render_to_surface(self.surface, values, self.value_type == 2)
        pygame.draw.line(self.surface, (255, 255, 255), (0, self.game_size[1]), (self.game_size[0], self.game_size[1]))
        for i, (draw_fn, _) in enumerate(self.buttons):
            text = draw_fn()
            rect = self.font.get_rect(text)
            rect.center = (i + 0.5) * self.button_size[0], self.game_size[1] + self.button_size[1] // 2
            self.font.render_to(self.surface, rect, None, (255, 255, 255))
            pygame.draw.line(self.surface, (255, 255, 255), ((i + 1) * self.button_size[0], self.game_size[1]),
                             ((i + 1) * self.button_size[0], self.game_size[1] + self.button_size[1]))
        self.screen.blit(self.surface)
        pygame.display.flip()

    def run(self):
        while True:
            if not pygame.event.peek() and not self.history[self.current].game.outcome():
                if self.players[self.history[self.current].game.to_play]:
                    self.move(self.player.play(self.history[self.current].game))
                    time.sleep(0.1)
                    continue
            event = pygame.event.wait()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if (action := self.history[self.current].game.mouse_click_to_action(event.pos)) is not None:
                    self.move(action)
                elif event.pos[1] > self.game_size[1]:
                    button = event.pos[0] // self.button_size[0]
                    self.buttons[button][1](event.button)
                    self.render()
            if event.type == pygame.WINDOWEXPOSED:
                self.render()
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                exit()


if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("player", type=str, help="The player module")
    parser.add_argument("--game", default="az_quiz", type=str, help="Game to evaluate")
    parser.add_argument("--mcts_simulations", default=100, type=int, help="MCTS simulations")
    parser.add_argument("--seed", default=None, type=int, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    analyzer = Analyzer(BoardGame.from_name(args.game)(), load_player(args, args.player), args.mcts_simulations)
    analyzer.run()
