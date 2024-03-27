### Assignment: cart_pole_pixels
#### Date: Deadline: Apr 09, 22:00
#### Points: 3 points + 4 bonus

The supplied [cart_pole_pixels_environment.py](https://github.com/ufal/npfl139/tree/master/labs/06/cart_pole_pixels_environment.py)
generates a pixel representation of the `CartPole` environment
as an $80×80$ `np.uint8` image with three channels, with each channel representing one time step
(i.e., the current observation and the two previous ones).

During evaluation in ReCodEx, three different random seeds will be employed,
each with time limit of 10 minutes, and if you reach an average return at least
450 on all of them, you obtain 3 points. The task is also
a [_competition_](https://ufal.mff.cuni.cz/courses/npfl139/2324-summer#competitions),
and at most 4 points will be awarded according to relative ordering of your
solutions.

The [cart_pole_pixels.py](https://github.com/ufal/npfl139/tree/master/labs/06/cart_pole_pixels.py)
template parses several parameters and creates the environment.
You are again supposed to train the model beforehand and submit
only the trained neural network.
