### Assignment: cheetah
#### Date: Deadline: Apr 30, 22:00
#### Points: 2 points

In this exercise, use the DDPG/TD3/SAC algorithm to solve the
[HalfCheetah environment](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).
If you start with DDPG, implementing the TD3 improvements
should make the hyperparameter search significantly easier.
However, for me, the SAC algorithm seems to work the best.

The template [cheetah.py](https://github.com/ufal/npfl139/tree/master/labs/09/cheetah.py)
only creates the environment and shows the evaluation in ReCodEx.

In ReCodEx, you are expected to submit an already trained model, which is
evaluated with two seeds, each for 100 episodes with a time limit of 10 minutes.
If your average return is at least 8000 on all of them, you obtain 2 points.
