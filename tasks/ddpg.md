### Assignment: ddpg
#### Date: Deadline: Apr 23, 22:00
#### Points: 5 points

Solve the continuous [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/)
and [InvertedDoublePendulum-v5](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)
environments using deep deterministic policy gradient algorithm.

Your goal is to reach an average return of -200 for `Pendulum-v1` and 9000 for `InvertedDoublePendulum-v5`
during 100 evaluation episodes.

Start with the [ddpg.py](https://github.com/ufal/npfl139/tree/past-2324/labs/08/ddpg.py)
template, which provides a simple network implementation in PyTorch. Feel free
to use TensorFlow ([ddpg.tf.py](https://github.com/ufal/npfl139/tree/past-2324/labs/08/ddpg.tf.py))
or JAX instead, if you like.

During evaluation in ReCodEx, two different random seeds will be employed for
both environments, and you need to reach the required return on all of them.
Time limit for each test is 10 minutes.
