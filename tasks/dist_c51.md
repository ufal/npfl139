### Assignment: dist_c51
#### Date: Deadline: Apr 09, 22:00
#### Points: 3 points

Extend the `q_network` assignment by solving the continuous
[CartPole-v1 environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
from the [Gymnasium library](https://gymnasium.farama.org/) using distributed
reinforcement learning algorithm C51.

Start with the [dist_c51.py](https://github.com/ufal/npfl139/tree/master/labs/05/dist_c51.py)
template. In the template, you must implement the `Network.compute_loss`
method, which constitutes the core of the C51 algorithm. In ReCodEx, the first
two tests verify your implementation by comparing the results to the reference
ones. You can also run two comparison tests present in the template locally by
using the `--verify` option.

Using the `Network.compute_loss` method, you should finish the C51
implementation. In the third test, ReCodEx verifies in the usual way that your
agent reaches an average return of 450 during 100 evaluation episodes.
