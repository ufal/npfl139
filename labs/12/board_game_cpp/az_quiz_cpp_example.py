#!/usr/bin/env python3
import numpy as np

import npfl139
npfl139.require_version("2425.12")
from npfl139.board_games import AZQuiz
import board_game_cpp

def evaluate(boards):
    # Boards have shape `[batch_size, AZQuiz::N, AZQuiz::N, AZQuiz::C]`.
    # You should return a pair of numpy arrays:
    # - prior policy with shape `[batch_size, AZQuiz::ACTIONS]`,
    # - estimated value function with shape `[batch_size]`.

    # In this example, return uniform policies and values of 0.
    return np.ones([boards.shape[0], AZQuiz.ACTIONS]) / AZQuiz.ACTIONS, np.zeros([boards.shape[0]])

# Run MCTS
game = AZQuiz()
board_game_cpp.select_game("az_quiz")
policy = board_game_cpp.mcts(game.board, game.to_play, evaluate, num_simulations=100, epsilon=0.25, alpha=0.3)
assert policy.shape == (AZQuiz.ACTIONS,)

# Start generating synthetic games
board_game_cpp.simulated_games_start(threads=2, num_simulations=100, sampling_moves=10, epsilon=0.25, alpha=0.3)
game = board_game_cpp.simulated_game(evaluate)
for board, policy, value in game:
    print(f"Board of shape {board.shape}, value {value}, policy {policy}")
board_game_cpp.simulated_games_stop()
