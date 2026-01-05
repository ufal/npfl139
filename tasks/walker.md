### Assignment: walker
#### Date: Deadline: Apr 30, 22:00
#### Points: 5 points

In this exercise we explore continuous robot control
by solving the continuous [BipedalWalker-v3 environment](https://gymnasium.farama.org/environments/box2d/bipedal_walker/).

**Note that the penalty of `-100` on crash makes the training considerably slower.
Even if all of DDPG, TD3 and SAC can be trained with original rewards, overriding
the reward at the end of episode to `0` speeds up training considerably.**

In ReCodEx, you are expected to submit an already trained model,
which is evaluated with two seeds, each for 100 episodes with a time
limit of 10 minutes. If your average return is at least 200, you obtain
5 points.

The [walker.py](https://github.com/ufal/npfl139/tree/past-2425/labs/08/walker.py)
template contains the skeleton for implementing the SAC agent, but you can
also solve the assignment with DDPG/TD3.
