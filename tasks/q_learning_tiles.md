### Assignment: q_learning_tiles
#### Date: Deadline: Mar 19, 22:00
#### Points: 3 points

Improve the `q_learning` task performance on the
[MountainCar-v0 environment](https://gymnasium.farama.org/environments/classic_control/mountain_car/)
using linear function approximation with tile coding.
Your goal is to reach an average reward of -110 during 100 evaluation episodes.

The environment methods are described in the `q_learning` assignment, with
the following changes:
- The `state` returned by the `env.step` method is a _list_ containing weight
  indices of the current state (i.e., the feature vector of the state consists
  of zeros and ones, and only the indices of the ones are returned). The
  action-value function is therefore approximated as a sum of the weights whose
  indices are returned by `env.step`.
- The `env.observation_space.nvec` returns a list, where the $i$-th element
  is a number of weights used by first $i$ elements of `state`. Notably,
  `env.observation_space.nvec[-1]` is the total number of the weights.

You can start with the [q_learning_tiles.py](https://github.com/ufal/npfl139/tree/past-2324/labs/03/q_learning_tiles.py)
template, which parses several useful parameters and creates the environment.
Implementing Q-learning is enough to pass the assignment, even if both N-step
Sarsa and Tree Backup converge a little faster. The default number of tiles in
tile encoding (i.e., the size of the list with weight indices) is
`args.tiles=8`, but you can use any number you want (but the assignment is
solvable with 8).

During evaluation in ReCodEx, three different random seeds will be employed, and
you need to reach the required return on all of them. The time limit for each
test is 5 minutes.
