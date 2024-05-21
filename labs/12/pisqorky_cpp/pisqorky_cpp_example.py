#!/usr/bin/env python3
import numpy as np

from pisqorky import Pisqorky
import pisqorky_cpp

def evaluate(boards):
    # Boards have shape `[batch_size, Pisqorky::N, Pisqorky::N, Pisqorky::C]`.
    # You should return a pair of numpy arrays:
    # - prior policy with shape `[batch_size, Pisqorky::ACTIONS]`,
    # - estimated value function with shape `[batch_size]`.

    # In this example, return uniform policies and values of 0.
    return np.ones([boards.shape[0], Pisqorky.ACTIONS]) / Pisqorky.ACTIONS, np.zeros([boards.shape[0]])

# Run MCTS
game = Pisqorky()
policy = pisqorky_cpp.mcts(game.board_internal, game.to_play, evaluate, num_simulations=100, epsilon=0.25, alpha=0.3)
assert policy.shape == (Pisqorky.ACTIONS,)

# Run the heuristic
action = pisqorky_cpp.heuristic(board=game.board_internal, to_play=game.to_play)
assert 0 <= action < Pisqorky.ACTIONS
print(f"Heuristic chose the action {action}")

# Start generating synthetic games
pisqorky_cpp.simulated_games_start(threads=2, num_simulations=100, sampling_moves=10, epsilon=0.25, alpha=0.3)
game = pisqorky_cpp.simulated_game(evaluate)
for board, policy, value in game:
    print(f"Board of shape {board.shape}, value {value}, policy {policy}")
pisqorky_cpp.simulated_games_stop()
