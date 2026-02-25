### Assignment: q_learning
#### Date: Deadline: Mar 11, 22:00
#### Points: 4 points

Solve the discretized [MountainCar-v0 environment](https://gymnasium.farama.org/environments/classic_control/mountain_car/)
from the [Gymnasium library](https://gymnasium.farama.org/) using the Q-learning
reinforcement learning algorithm. Note that this task still does not require
PyTorch.

The environment methods and properties are described in the `monte_carlo` assignment.
Once you finish training (which you indicate by passing `start_evaluation=True`
to `reset`), your goal is to reach an average return of -150 during 100
evaluation episodes.

You can start with the [q_learning.py](https://github.com/ufal/npfl139/tree/master/labs/02/q_learning.py)
template, which parses several useful parameters, creates the environment
and illustrates the overall usage. Note that setting hyperparameters of
Q-learning is a bit tricky – I usually start with a larger value of $ε$ (like 0.2
or even 0.5) and then gradually decrease it to almost zero.

During evaluation in ReCodEx, three different random seeds will be employed, and
you need to reach the required return on all of them. The time limit for each
test is 5 minutes.
