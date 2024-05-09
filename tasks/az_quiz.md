### Assignment: az_quiz
#### Date: Deadline: May 21, 22:00
#### Points: 10 points + 5 bonus

In this competition assignment, use Monte Carlo Tree Search to learn
an agent for a simplified version of [AZ-kvíz](https://cs.wikipedia.org/wiki/AZ-kv%C3%ADz).
In our version, the agent does not have to answer questions and we assume
that **all answers are correct**.

The game itself is implemented in the
[az_quiz.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz.py)
module (using the default `randomized=False` constructor argument).

The evaluation in ReCodEx should be implemented by returning an object
implementing a method `play`, which given an AZ-kvíz instance returns the chosen
move. The illustration of the interface is in the
[az_quiz_player_random.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz_player_random.py)
module, which implements a random agent.

Your solution in ReCodEx is automatically evaluated against a very simple heuristic
[az_quiz_player_simple_heuristic.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz_player_simple_heuristic.py),
playing 56 games as a starting player and 56 games as a non-starting player. The
time limit for the games is 10 minutes and you should see the win rate directly in
ReCodEx. If you achieve at least 95%, you will pass the assignment.
A better heuristic [az_quiz_player_fork_heuristic.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz_player_simple_heuristic.py)
is also available for your evaluations.

The final competition evaluation will be performed after the deadline by
a round-robin tournament. In this tournament, we also consider games
where the first move is chosen for the first player (`FirstChosen` label
in ReCodEx, `--first_chosen` option of the evaluator).

The [az_quiz_evaluator.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz_evaluator.py)
can be used to evaluate any two given implementations and there are two
interactive players available,
[az_quiz_player_interactive_mouse.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz_player_interactive_mouse.py)
and [az_quiz_player_interactive_keyboard.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz_player_interactive_keyboard.py).

The starting PyTorch template is available in the
[az_quiz_agent.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz_agent.py)
module; TensorFlow version [az_quiz_agent_tf.py](https://github.com/ufal/npfl139/tree/master/labs/11/az_quiz_agent_tf.py)
is available too.

**To get regular points, you must implement an AlphaZero-style algorithm.
However, any algorithm can be used in the competition.**
