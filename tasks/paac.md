### Assignment: paac
#### Date: Deadline: Apr 23, 22:00
#### Points: 3 points

Solve the [LunarLander-v3 environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
with continuous observations using parallel actor-critic algorithm, employing
the vectorized environment described in the `car_racing` assignment.

Your goal is to reach an average return of 250 during 100 evaluation episodes.

Start with the [paac.py](https://github.com/ufal/npfl139/tree/master/labs/07/paac.py)
template, which provides a simple network implementation in PyTorch, support for
saving a trained agent, and ReCodEx evaluation of a submitted agent. The used
environment is configurable, so you can experiment also with other environments;
for example, I would first solve the `CartPole` environment before progressing
to `LunarLander`.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.
