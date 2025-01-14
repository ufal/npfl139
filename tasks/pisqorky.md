### Assignment: pisqorky
#### Date: Deadline: May 21, 22:00
#### Points: 5 points + 5 bonus; either this or `az_quiz_randomized` is required for automatically passing the exam

Train an agent on [Piškvorky](https://cs.wikipedia.org/wiki/Pi%C5%A1kvorky),
usually called [Gomoku](https://en.wikipedia.org/wiki/Gomoku) internationally.

Because the game is more complex than `az_quiz`, you probably have to use the
C++ template [pisqorky_cpp](https://github.com/ufal/npfl139/tree/past-2324/labs/12/pisqorky_cpp).
Note that the template shares a lot of code with `az_quiz_cpp`; it would be
definitely better to refactor it to use the `BoardGame` ancestor and to share
the common functionality.

The C++ template also provides quite a strong heuristic; in ReCodEx, your agent
is evaluated against it, and if it reaches at least 25% win
rate in 100 games (50 as a starting player and 50 as a non-starting player),
you get the regular points. The final competition evaluation will be
performed after the deadline by a round-robin tournament.

**To get regular points, you must implement an AlphaZero-style algorithm.
However, any algorithm can be used in the competition.**
