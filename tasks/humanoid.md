### Assignment: humanoid
#### Date: Deadline: May 14, 22:00
#### Points: 3 points; not required for passing the exam with grade 1 by solving all assignments

In this exercise, use the DDPG/TD3/SAC algorithm to solve the
[Humanoid environment](https://gymnasium.farama.org/environments/mujoco/humanoid/).

The template [humanoid.py](https://github.com/ufal/npfl139/tree/master/labs/09/humanoid.py)
only creates the environment and shows the evaluation in ReCodEx.

In ReCodEx, you are expected to submit an already trained model, which is
evaluated with two seeds, each for 100 episodes with a time limit of 10 minutes.
If your average return is at least 8000 on all of them, you obtain 3 points.
