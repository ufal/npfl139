### Assignment: atari_gamer
#### Date: Deadline: June 30, 22:00
#### Points: 4 points

In this long-term assignment, your goal is to solve one of the
[Atari games](https://ale.farama.org/environments/). Currently, you can choose
one of the following games:
- [Pong](https://ale.farama.org/environments/pong/) (one of the easiest games),
  where you must surpass the average return of 16 in 10 episodes
- [Breakout](https://ale.farama.org/environments/breakout/) (a more difficult
  one game), where you must surpass the average return of 200 in 10 episodes.

If you would like to train an agent for a different Atari game, write us on
Piazza, and we will decide the required score and add it to the list. Note that
you can play any Atari game interactively with WASD and SPACE using the
`python3 -m npfl139.envs.atari_interactive GAME_NAME [--zoom=4] [--frame_skip=1]`
command, so for example `python3 -m npfl139.envs.atari_interactive Pong`.

The template [atari_gamer.py](https://github.com/ufal/npfl139/tree/past-2425/labs/06/atari_gamer.py)
shows how to create the Atari environment. The `args.game` is the chosen game
and `args.frame_skip` option specified the required frame skipping (these
options are used also in ReCodEx to construct the evaluation environment), and
additional wrappers can be applied to the environment at the beginning of the
`main` method.

While you can use any algorithm from the lecture to solve the environment, some
distributional DQN approach (C51/QR-DQN/IQN) is a reasonable choice (as is the
PPO algorithm). The time limit in ReCodEx is 15 minutes.
