### Assignment: az_quiz
#### Date: Deadline: May 21, 22:00
#### Points: 5 points + 5 bonus

In this competition assignment, use Monte Carlo Tree Search to learn
an agent for a simplified version of [AZ-kvíz](https://cs.wikipedia.org/wiki/AZ-kv%C3%ADz).
In our version, the agent does not have to answer questions and we assume
that **all answers are correct**.

The game itself is implemented in the
[az_quiz](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/az_quiz.py)
module, which is a subclass of a general
[board_game](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/board_game.py).

The evaluation in ReCodEx is performed via the
[board_game_player](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/board_game_player.py)
interface, most notably through the `play` method, which given an AZ-kvíz
instance returns the chosen move. The illustration of the interface is in the
[az_quiz_player_random](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/az_quiz_player_random.py)
module, which implements a random agent.

Your solution in ReCodEx is evaluated against a very simple heuristic
[az_quiz_player_simple_heuristic](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/az_quiz_player_simple_heuristic.py),
in two different settings: first playing 560 games (280 as a starting player
and 280 games as a non-starting player) with a total limit of 25 second,
and then playing just 112 games with a total limit of 5 minutes. During
the first test, you are expected to use **just the trained policy** to fulfil
the time limit; in the second test, you can use also MCTS during evaluation.
In all evaluations, you can see the win rate of your agent directly in ReCodEx,
and to pass you need to achieve at least 95% in both tests. To distinguish
the two mentioned tests in ReCodEx, `num_simulations` argument is set to 0
during the first test, while it is not modified in any other tests.

ReCodEx also evaluates your agent against several more advanced players:
a publicly available better heuristic
[az_quiz_player_fork_heuristic](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/az_quiz_player_simple_heuristic.py)
and three neural network agents (which you do not have access to apart
from the ReCodEx evaluation). These additional evaluations also consist
of 112 games with a time limit of 5 minutes, and are provided just for
your convenience.

The final competition evaluation will be performed after the deadline by
a round-robin tournament. In this tournament, we also consider games
where the first move is chosen for the first player (`FirstChosen` label
in ReCodEx, `--first_chosen` option of the evaluator).

The `evaluate` method of the
[board_game_evaluator](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/board_game_evaluator.py)
can be used in your code to evaluate any two given players. Furthermore, you can
evaluate players also from the command line using the
[npfl139.board_games.evaluate](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/evaluate.py)
module. As players, you can use either the provided players
[random](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/az_quiz_player_random.py),
[simple_heuristic](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/az_quiz_player_simple_heuristic.py),
[fork_heuristic](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/az_quiz_player_fork_heuristic.py),
[mouse](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/az_quiz_player_mouse.py), and
[keyboard](https://github.com/ufal/npfl139/blob/master/labs/npfl139/board_games/az_quiz_player_keyboard.py), or
you can pass the name of any module implementing a `Player: BoardGamePlayer`
class. For illustration, you can use `python3 -m npfl139.board_games.evaluate
--render mouse fork_heuristic` to interactively play against the fork heuristic, or
`python3 -m npfl139.board_games.evaluate az_quiz_agent.py:--model_path=PATH:--num_simulations=0 simple_heuristic`
to evaluate `az_quiz_agent.py` with the specified arguments against the simple heuristic.

The template for this assignment is available in the
[az_quiz_agent.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz_agent.py)
module, or in a variant [board_game_agent.py](https://github.com/ufal/npfl139/tree/master/labs/11/board_game_agent.py).
The latter is nearly identical, but it is slightly more general, not
mentioning `AZQuiz` directly in the code; instead, the game (and the player to
evaluate against) is specified only in an argument.

**To get regular points, you must implement an AlphaZero-style algorithm.
However, any algorithm can be used in the competition.**
