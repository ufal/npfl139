### Assignment: pisqorky
#### Date: Deadline: Jun 30, 22:00
#### Points: 5 points + 5 bonus; either this or `az_quiz_randomized` is required for automatically passing the exam

**Note that the environment implementation is still being finalized,
it will be released in a few days.**

Train an agent on [Piškvorky](https://cs.wikipedia.org/wiki/Pi%C5%A1kvorky),
usually called [Gomoku](https://en.wikipedia.org/wiki/Gomoku) internationally.

Because the game is more complex than `az_quiz`, you probably have to use the
C++ template [board_game_cpp](https://github.com/ufal/npfl139/tree/master/labs/12/board_game_cpp).

The game provides quite a strong heuristic; in ReCodEx, your agent is evaluated
against it, and if it reaches at least 25% win rate in 100 games (50 as
a starting player and 50 as a non-starting player), you get the regular points.
The final competition evaluation will be performed after the deadline by
a round-robin tournament.

**To get regular points, you must implement an AlphaZero-style algorithm.
However, any algorithm can be used in the competition.**
