### Assignment: ppo
#### Date: Deadline: ~~May 7~~ May 14, 22:00
#### Points: 4 points

Implement the PPO algorithm in a single-agent settings. Notably, solve
the `SingleCollect` environment implemented by the
[multi_collect_environment.py](https://github.com/ufal/npfl139/tree/past-2324/labs/10/multi_collect_environment.py)
module. To familiarize with it, you can [watch a trained agent](https://ufal.mff.cuni.cz/~straka/courses/npfl139/2324/videos/single_collect.mp4)
and you can run the module directly, controlling the agent with the arrow keys.
In the environment, your goal is to reach a known place, obtaining rewards
based on the agent's distance. If the agent is continuously occupying the place
for some period of time, it gets a large reward and the place is moved randomly.
The environment runs for 250 steps and it is considered solved if you obtain
a return of at least 500.

The [ppo.py](https://github.com/ufal/npfl139/tree/past-2324/labs/10/ppo.py)
PyTorch template contains a skeleton of the PPO algorithm implementation.
TensorFlow template [ppo.tf.py](https://github.com/ufal/npfl139/tree/past-2324/labs/10/ppo.tf.py) is also available.
Regarding the unspecified hyperparameters, I would consider the following ranges:
- `batch_size` between 64 and 512
- `clip_epsilon` between 0.1 and 0.2
- `epochs` between 1 and 10
- `gamma` between 0.97 and 1.0
- `trace_lambda` is usually 0.95
- `envs` between 16 and 128
- `worker_steps` between tens and hundreds

My implementation trains in approximately three minutes of CPU time.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the average return of 450 on all of them. Time limit for each test
is 10 minutes.
