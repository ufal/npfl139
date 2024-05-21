#!/usr/bin/env python3
import numpy as np

from az_quiz import AZQuiz
import az_quiz_cpp

def evaluate(boards):
    # Boards have shape `[batch_size, AZQuiz::N, AZQuiz::N, AZQuiz::C]`.
    # You should return a pair of numpy arrays:
    # - prior policy with shape `[batch_size, AZQuiz::ACTIONS]`,
    # - estimated value function with shape `[batch_size]`.

    # In this example, return uniform policies and values of 0.
    return np.ones([boards.shape[0], AZQuiz.ACTIONS]) / AZQuiz.ACTIONS, np.zeros([boards.shape[0]])

# Run MCTS
game = AZQuiz()
policy = az_quiz_cpp.mcts(game.board_internal, game.to_play, game._randomized, evaluate, num_simulations=100, epsilon=0.25, alpha=0.3)
assert policy.shape == (AZQuiz.ACTIONS,)

# Start generating synthetic games
az_quiz_cpp.simulated_games_start(threads=2, randomized=False, num_simulations=100, sampling_moves=10, epsilon=0.25, alpha=0.3)
game = az_quiz_cpp.simulated_game(evaluate)
for board, policy, value in game:
    print(f"Board of shape {board.shape}, value {value}, policy {policy}")
az_quiz_cpp.simulated_games_stop()
