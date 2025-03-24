#!/usr/bin/env python3
import argparse
import collections

import numpy as np
from typing import Generic, TypeVar

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size to sample from the buffer")
parser.add_argument("--max_length", default=128, type=int, help="Maximum length of the replay buffer")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.

NamedTuple = TypeVar("NamedTuple")  # A generic type fulfilling the NamedTuple protocol.


class PrioritizedReplayBuffer(Generic[NamedTuple]):
    """A prioritized replay buffer with a limited capacity.

    The individual items must be named tuples of data convertible to Numpy
    arrays, each with a fixed shape. The whole replay buffer stores items in an
    efficient manner by keeping a single named tuple of Numpy arrays containing
    all the data."""
    def __init__(self, max_length: int) -> None:
        self._len: int = 0
        self._max_length: int = max_length
        self._offset: int = 0  # Used when the buffer is full and overwriting in a circular manner.
        self._data: NamedTuple | None = None

        # TODO: Create data structures for priorities.
        ...

    def __len__(self) -> int:
        return self._len

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def data(self) -> NamedTuple | None:
        return self._data

    def __getitem__(self, index: int | np.ndarray) -> NamedTuple:
        """Return the item or items at the given index or indices.

        Returns:
          A NamedTuple of Numpy arrays containing the item(s) at the given index(es).
        """
        return self._data._make(value[index] for value in self._data)

    def append(self, item: NamedTuple, priority: float | None = None) -> int:
        """Append a new item with an optional non-negative priority to the replay buffer.

        If the buffer is empty, it is allocated to full capacity using Numpy
        arrays of shapes and data types of the provided item.

        The priority must be non-negative. If no priority is provided, use the
        largest priority ever seen, using 1.0 if no priority has been set yet.

        Returns:
          The index at which the item was stored.
        """
        # Allocate the buffer on the first append.
        if not self._len:
            values = [np.empty((self._max_length, *value.shape), dtype=value.dtype) for value in map(np.asarray, item)]
            self._data = item._make(values)

        # Select the index to store the new item, updating the sizes.
        if self._len < self._max_length:
            index, self._len = self._len, self._len + 1
        else:
            index, self._offset = self._offset, (self._offset + 1) % self._max_length

        # Store the new item on the given index.
        for i, value in enumerate(item):
            self._data[i][index] = value

        # Store the priority and perform required updates.
        self.update_priority(index, priority)

        return index

    def update_priority(self, index: int, priority: float | None = None) -> None:
        """Update the priority of an item in the replay buffer.

        The priority must be non-negative. If no priority is provided, use the
        largest priority ever seen, using 1.0 if no priority has been set yet.
        """
        assert 0 <= index < self._len
        # TODO: Store the priority and perform required updates.
        ...

    def sample(self, size: int, generator=np.random) -> tuple[NamedTuple, np.ndarray, np.ndarray]:
        """Sample a batch of items from the replay buffer.

        The commulative probabilities of the items to sample are computed using
        `(generator.uniform(size=size) + np.arange(size)) / size`. This way,
        there is a single sample in every 1/size interval, decreasing the change
        of sampling the same item multiple times.

        Returns:
          A triple containing the sampled items, their indices, and their (normalized) probabilities.
        """
        samples = (generator.uniform(size=size) + np.arange(size)) / size

        # TODO: Generate the sampled items so that the i-th sampled item fulfills:
        # - the sum of probabilities of items preceeding the sampled item in the buffer is <= sample[i],
        # - the sum of probabilities of the above items plus the sampled item is > sample[i].
        indices: np.ndarray = ...

        # TODO: Compute the probabilities of the selected indices.
        probabilities: np.ndarray = ...

        return self[indices], indices, probabilities


def main(args: argparse.Namespace) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(args.max_length)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    if main_args.max_length & (main_args.max_length - 1):
        raise ValueError("For the tests to work, max_length must be a power of two.")

    buffer = main(main_args)

    # Run a random test that verifies the required behavior of the buffer.
    Element = collections.namedtuple("Element", ["value"])
    values = np.zeros(main_args.max_length, dtype=np.int64)
    priorities = np.zeros(main_args.max_length + 1, dtype=np.float64)

    for i in range(3 * main_args.max_length):
        # Either append a new character, of update priority of an existing one.
        if i > main_args.max_length and i % 3 == 0:
            buffer.update_priority(j := (i % main_args.max_length), i if i % 2 else None)
        else:
            j = buffer.append(Element(i), i if i % 2 else None)
            values[j] = i
        priorities[j + 1] = i if i % 2 else max(i - 1, 1)

        # Sample from the replay buffer if it is large enough.
        if i >= main_args.batch_size:
            items, indices, probs = buffer.sample(main_args.batch_size, np.random.RandomState(main_args.seed + i))

            # Generate the same samples and compute the buffer probabilities.
            samples = (np.random.RandomState(main_args.seed + i).uniform(size=main_args.batch_size)
                       + np.arange(main_args.batch_size)) / main_args.batch_size

            priorities_cumsum = np.cumsum(priorities)

            # Final verification.
            assert np.all(priorities_cumsum[indices] <= samples * priorities_cumsum[-1])
            assert np.all(priorities_cumsum[indices + 1] > samples * priorities_cumsum[-1])
            np.testing.assert_allclose(probs * priorities_cumsum[-1], priorities[indices + 1], atol=1e-5)
            np.testing.assert_equal(items.value, values[indices])

    print("All checks passed.")
