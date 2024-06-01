### Assignment: memory_game
#### Date: Deadline: Jun 28, 22:00
#### Points: 3 points

In this exercise we explore a partially observable environment.
Consider a one-player variant of a memory game (pexeso), where a player repeatedly
flip cards. If the player flips two cards with the same symbol in succession,
the cards are removed and the player recieves a reward of +2. Otherwise the
player recieves a reward of -1. An episode ends when all cards are removed.
Note that it is valid to try to flip an already removed card.

Let there be $N$ cards in the environment, $N$ being even. There are $N+1$
actions – the actions $1..N$ flip the corresponding card, and the action 0
flips the unused card with the lowest index (or the card $N$ if all have
been used already). The observations consist of a pair of discrete values
_(card, symbol)_, where the _card_ is the index of the card flipped, and
the _symbol_ is the symbol on the flipped card; the `env.observation_space.nvec`
is a pair $(N, N/2)$, representing there are $N$ card indices and $N/2$
symbol indices.

Every episode can be ended by at most $3N/2$ actions, and the required
return is therefore greater or equal to zero. Note that there is a limit
of at most $2N$ actions per episode. The described environment is provided
by the [memory_game_environment.py](https://github.com/ufal/npfl139/tree/master/labs/14/memory_game_environment.py)
module.

Your goal is to solve the environment, using supervised learning via the provided
_expert episodes_ and networks with external memory. The environment implements
an `env.expert_episode()` method, which returns a fresh correct episode
as a list of `(state, action)` pairs (with the last `action` being `None`).

ReCodEx evaluates your solution on environments with 8, 12 and 16 cards
(utilizing the `--cards` argument). For each card number, 100 episodes are
simulated once you pass `evaluating=True` to `env.reset` and your solution gets
1 point if the average return is nonnegative. You can
train the agent directly in ReCodEx (the time limit is 15 minutes),
or submit a pre-trained one.

PyTorch template [memory_game.py](https://github.com/ufal/npfl139/tree/master/labs/14/memory_game.py)
shows a possible way to use memory augmented networks. TensorFlow template
[memory_game.tf.py](https://github.com/ufal/npfl139/tree/master/labs/14/memory_game.tf.py)
is also available.
