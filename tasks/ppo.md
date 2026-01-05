### Assignment: ppo
#### Date: Deadline: May 7, 22:00
#### Points: 5 points

Use the PPO algorithm to solve the `SingleCollect` environment implemented by the
[single_collect.py](https://github.com/ufal/npfl139/tree/past-2425/labs/npfl139/envs/single_collect.py)
module. To familiarize with it, you can [watch a trained agent](https://ufal.mff.cuni.cz/~straka/courses/npfl139/2425/videos/single_collect.mp4)
and you can play it interactively using `python3 -m npfl139.envs.single_collect_interactive`,
controlling the agent with the arrow keys. In the environment, your goal is to
reach a known place (e.g., a _food source_), obtaining rewards based on the
agent's distance. If the agent is continuously occupying the place for some
period of time, it gets a large reward and the place is moved randomly. The
environment runs for 250 steps and it is considered solved if you obtain
a return of at least 500.

The [ppo.py](https://github.com/ufal/npfl139/tree/past-2425/labs/10/ppo.py)
template contains a skeleton of the PPO algorithm implementation.
My implementation trains in approximately three minutes of CPU time.

During evaluation in ReCodEx, two different random seeds will be employed, and
you need to reach the average return of 500 on all of them. Time limit for each test
is 10 minutes.
