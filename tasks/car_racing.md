### Assignment: car_racing
#### Date: Deadline: Mar 26, 22:00
#### Points: 6 points + 6 bonus

The goal of this competition is to use Deep Q Networks (and any of Rainbow improvements)
on a more real-world [CarRacing-v2 environment](https://gymnasium.farama.org/environments/box2d/car_racing/)
from the [Gymnasium library](https://gymnasium.farama.org/).

The supplied [car_racing_environment.py](https://github.com/ufal/npfl139/tree/past-2324/labs/04/car_racing_environment.py)
provides the environment. The states are RGB `np.uint8` images of size
$96×96×3$, but you can downsample them even more. The actions
are also continuous and consist of an array with the following three elements:
- `steer` in range [-1, 1]
- `gas` in range [0, 1]
- `brake` in range [0, 1]; note that full brake is quite aggressive, so you
  might consider using less force when braking
Internally you should probably generate discrete actions and convert them to the
required representation before the `step` call. Alternatively, you might set
`args.continuous=0`, which changes the action space from continuous to 5 discrete
actions – do nothing, steer left, steer right, gas, and brake. But you can
experiment with different action space if you want.

The environment also supports frame skipping (`args.frame_skipping`), which
improves its performance (only some frames need to be rendered). Note that
ReCodEx respects both `args.continuous` and `args.frame_skipping` during
evaluation.

In ReCodEx, you are expected to submit an already trained model,
which is evaluated on 15 different tracks with a total time
limit of 15 minutes. If your average return is at least 300, you obtain
6 points. The task is also a [_competition_](https://ufal.mff.cuni.cz/courses/npfl139/2324-summer#competitions),
and at most 6 points will be awarded according to relative ordering of your
solutions.

The [car_racing.py](https://github.com/ufal/npfl139/tree/past-2324/labs/04/car_racing.py)
template parses several useful parameters and creates the environment.
Note that the [car_racing_environment.py](https://github.com/ufal/npfl139/tree/past-2324/labs/04/car_racing_environment.py)
can be executed directly and in that case you can drive the car using arrows.

Also, you might want to use a **vectorized version of the environment** for
training, which runs several individual environments in separate processes.
The template contains instructions how to create it. The vectorized environment
expects a vector of actions and returns a vector of observations, rewards, dones
and infos. When one of the environments finishes, it is **automatically reset**
in the next step.
