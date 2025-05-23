### Assignment: pisqorky
#### Date: Deadline: Jun 30, 22:00
#### Points: 5 points + 5 bonus; either this or `az_quiz_randomized` is required for automatically passing the exam

Train an agent on [Piškvorky](https://cs.wikipedia.org/wiki/Pi%C5%A1kvorky),
usually called [Gomoku](https://en.wikipedia.org/wiki/Gomoku) internationally,
implemented in the [pisqorky](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/pisqorky.py)
module.

Because the game is more complex than `az_quiz`, you probably have to use the
C++ template [board_game_cpp](https://github.com/ufal/npfl139/tree/master/labs/12/board_game_cpp).
That template now supports both `AZQuiz` and `Pisqorky`.

The game provides quite a strong heuristic called simply `heuristic`;
in ReCodEx, your agent is evaluated against it, again in two different settings:
first playing 100 games (50 as a starting player and 50 as a non-starting
player) with a total limit of 15 minutes, and then playing just 10 games with
a total limit of again 15 minutes. During the first test, you are expected to
again use just the trained policy (`num_simulations` is set to 0), while
in the second test you might use MCTS if you want (`num_simulations` is
not modified). In order to pass, you need to achieve at least 50% win rate
in both tests.

ReCodEx also evaluates your agent against a single neural network agent.
The evaluation consists of 100 games (50 starting, 50 non-starting) in
`FirstChosen` setting without MCTS (`num_simulations` is set to 0),
with time limit of 15 minutes.

The final competition evaluation will be performed after the deadline by
a round-robin tournament, when the first move is chosen for the first player.

**To get regular points, you must implement an AlphaZero-style algorithm.
However, any algorithm can be used in the competition.**
