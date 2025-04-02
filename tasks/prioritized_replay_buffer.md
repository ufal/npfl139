### Assignment: prioritized_replay_buffer
#### Date: Deadline: Apr 09, 22:00
#### Points: 2 points
#### Tests: prioritized_replay_buffer_tests

In this assignment, your goal is to implement an efficient prioritized replay
buffer with logarithmic complexity of sampling. Start with the
[prioritized_replay_buffer.py](https://github.com/ufal/npfl139/tree/master/labs/05/prioritized_replay_buffer.py)
template, which contains the skeleton of the implementation and a detailed
description of the functionality that you must implement.

Note that compared to
[npfl139.ReplayBuffer](https://github.com/ufal/npfl139/blob/master/labs/npfl139/replay_buffer.py),
the template implementation is more memory efficient: the buffer elements must
be named tuples of data convertible to Numpy arrays with a constant shape,
and the whole replay buffer stores items in a single named tuple of Numpy
arrays containing all the data.

When executed directly, the template runs randomized tests verifying the
prioritized replay buffer behaves as expected. In ReCodEx, similar tests are
executed, and your performance must be at most twice the running time of the
reference solution.

In both local and ReCodEx tests, the maximum replay buffer capacity is always
a power of 2 (even if the reference implementation can handle an arbitrary
limit). Note that to avoid precision loss, you should store the priorities using
64-bit floats.

#### Tests Start: prioritized_replay_buffer_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 prioritized_replay_buffer.py --max_length=128`

2. `python3 prioritized_replay_buffer.py --max_length=8192 --batch_size=8`

3. `python3 prioritized_replay_buffer.py --max_length=131072 --batch_size=2`
#### Tests End:
