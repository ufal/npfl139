### Assignment: walker_hardcore
#### Date: Deadline: Apr 30, 22:00
#### Points: 5 points + 5 bonus

As an extension of the `walker` assignment, solve the
_hardcore_ version of the [BipedalWalker-v3 environment](https://gymnasium.farama.org/environments/box2d/bipedal_walker/).

**Note that the penalty of `-100` on crash can discourage or even stop training,
so overriding the reward at the end of episode to `0` (or descresing it
substantially) makes the training considerably easier (I have not surpassed
return `0` with neither TD3 nor SAC with the original `-100` penalty).**

In ReCodEx, you are expected to submit an already trained model,
which is evaluated with three seeds, each for 100 episodes with a time
limit of 10 minutes. If your average return is at least 100, you obtain
5 points. The task is also a [_competition_](https://ufal.mff.cuni.cz/courses/npfl139/2324-summer#competitions),
and at most 5 points will be awarded according to relative ordering of your
solutions.

The [walker_hardcore.py](https://github.com/ufal/npfl139/tree/past-2324/labs/09/walker_hardcore.py)
template shows a basic structure of evaluaton in ReCodEx, but
you most likely want to start either with [ddpg.py](https://github.com/ufal/npfl139/tree/past-2324/labs/08/ddpg.py).
or with [walker.py](https://github.com/ufal/npfl139/tree/past-2324/labs/09/walker.py)
and just change the `env` argument to `BipedalWalkerHardcore-v3`.
