### Assignment: q_network
#### Date: Deadline: Mar 26, 22:00
#### Points: 5 points

Solve the continuous [CartPole-v1 environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
from the [Gymnasium library](https://gymnasium.farama.org/) using Q-learning
with neural network as a function approximation.

You can start with the [q_network.py](https://github.com/ufal/npfl139/tree/past-2425/labs/04/q_network.py)
template, which provides a simple network implementation in PyTorch.

The continuous environment is very similar to a discrete one, except
that the states are vectors of real-valued observations with shape
`env.observation_space.shape`.

Use Q-learning with neural network as a function approximation, which for
a given state returns state-action values for all actions. You can use any
network architecture, but one hidden layer of several dozens ReLU units is a good start.
Your goal is to reach an average return of 450 during 100 evaluation episodes.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes (so you can train in ReCodEx, but you can also pretrain your
network if you like).
