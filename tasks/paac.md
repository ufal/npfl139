### Assignment: paac
#### Date: Deadline: Apr 23, 22:00
#### Points: 3 points

Solve the [CartPole-v1 environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
using parallel actor-critic algorithm, employing the vectorized
environment described in the `car_racing` assignment.

Your goal is to reach an average return of 450 during 100 evaluation episodes.

Start with the [paac.py](https://github.com/ufal/npfl139/tree/past-2324/labs/08/paac.py)
template, which provides a simple network implementation in PyTorch. Feel
free to use TensorFlow or JAX instead, if you like.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 10 minutes.
