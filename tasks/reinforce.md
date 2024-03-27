### Assignment: reinforce
#### Date: Deadline: Apr 09, 22:00
#### Points: 4 points

Solve the continuous [CartPole-v1 environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
from the [Gymnasium library](https://gymnasium.farama.org/) using the REINFORCE
algorithm.

Your goal is to reach an average return of 490 during 100 evaluation episodes.

Start with the [reinforce.py](https://github.com/ufal/npfl139/tree/master/labs/06/reinforce.py)
template, which provides a simple network implementation in PyTorch. Feel free
to use TensorFlow ([reinforce.tf.py](https://github.com/ufal/npfl139/tree/master/labs/06/reinforce.tf.py))
or JAX instead, if you like.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.
