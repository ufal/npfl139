### Assignment: lunar_lander
#### Date: Deadline: Mar 19, 22:00
#### Points: 5 points + 5 bonus

Solve the [LunarLander-v2 environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
from the [Gymnasium library](https://gymnasium.farama.org/) Note that this task
does not require PyTorch.

The environment methods and properties are described in the `monte_carlo` assignment,
but include one additional method:
- `expert_trajectory(seed=None) → trajectory`: This method generates one expert
  trajectory, where `trajectory` is a list of triples _(state, action, reward)_,
  where the _action_ and _reward_ is `None` when reaching the terminal state.
  If a `seed` is given, the expert trajectory random generator is reset before
  generating the trajectory.

  You **cannot** change the implementation of this method or use its internals in
  any other way than just calling `expert_trajectory()`. Furthermore,
  you can use this method only during training, **not** during evaluation.

To pass the task, you need to reach an average return of 0 during 1000 evaluation episodes.
During evaluation in ReCodEx, three different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 15 minutes.

The task is additionally a [_competition_](https://ufal.mff.cuni.cz/courses/npfl139/2425-summer#competitions),
and at most 5 points will be awarded according to the relative ordering of your
solutions.

You can start with the [lunar_lander.py](https://github.com/ufal/npfl139/tree/master/labs/03/lunar_lander.py)
template, which parses several useful parameters, creates the environment
and illustrates the overall usage.

In the competition, **you should consider the environment states meaning to be
unknown**, so you **cannot use the knowledge about how they are created**. But
you can learn any such information from the data.
