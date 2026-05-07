### Assignment: az_quiz_randomized
#### Date: Deadline: Jun 30, 22:00
#### Points: 5 points; either this or `pisqorky` is required for automatically passing the exam

Extend the `az_quiz` assignment to handle the possibility of wrong
answers. Therefore, when choosing a field (an action), you might not
claim it; in such a case, the state of the field becomes “failed”. When
a “failed” field is chosen as an action by a player, then either
- it is successfully claimed by the player (they “answer correctly”); or
- if the player “answers incorrectly”, the field is claimed by the opposing
  player; however, in this case, the original player continue playing
  (i.e., the players do not alternate in this case).

To instantiate this randomized game variant, either pass `randomized=True`
to the `npfl139.board_games.AZQuiz`, or use `az_quiz_randomized` as a board
games (e.g., as the argument to `npfl139.board_games.evaluate` or to
`npfl139.board_games.BoardGame.from_name`).

Your goal is to propose how to modify the Monte Carlo Tree Search to **properly
handle** stochastic MDPs. The information about distribution of possible next
states is provided by the `AZQuiz.all_moves` method, which returns a list of
`(probability, az_quiz_instance)` next states (in our environment, there are
always two possible next states).

Your implementation must be capable of training and achieving at least 90% win
rate against the simple heuristic. The evaluation is performed in the same
setting as in `az_quiz`, so in order to pass, you must achieve 90% win rate both
in the first test (560 games in 25 seconds with `num_simulations=0`) and in the
second test (112 games in 5 minutes without modifying `num_simulations`).

Additionally, part of this assignment is also to **write us
on Piazza** (once you pass in ReCodEx) a **description** of how you **handle the
stochasticity** in MCTS; you will get points only after we finish the discussion.
