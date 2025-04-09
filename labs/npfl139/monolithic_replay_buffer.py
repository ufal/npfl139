# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Generic, Sequence, TypeVar

import numpy as np

NamedTuple = TypeVar("NamedTuple")  # A generic type fulfilling the NamedTuple protocol.


class MonolithicReplayBuffer(Generic[NamedTuple]):
    """A monolithic replay buffer with a limited capacity.

    The individual items must be named tuples of data convertible to Numpy
    arrays, each with a fixed shape. The whole replay buffer stores items in an
    efficient manner by keeping a single named tuple of Numpy arrays containing
    all the data.

    Parameters:
      max_length: The maximum number of items the replay buffer can store.
      seed: An optional random seed for sampling; fresh one created if None.
    """
    def __init__(self, max_length: int, seed: int | None = None) -> None:
        assert isinstance(max_length, int), "The max_length argument must be an integer"
        assert max_length > 0, "The max_length argument must be a positive integer"

        self._len: int = 0
        self._max_length: int = max_length
        self._offset: int = 0  # Used when the buffer is full and overwriting in a circular manner.
        self._data: NamedTuple | None = None
        self._generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of items in the replay buffer."""
        return self._len

    @property
    def max_length(self) -> int:
        """Return the maximum capacity of the replay buffer."""
        return self._max_length

    @property
    def data(self) -> NamedTuple | None:
        """Return the data stored in the replay buffer as a named tuple of Numpy arrays."""
        return self._data

    def __getitem__(self, index: int | np.ndarray) -> NamedTuple:
        """Return the item or items at the given index or indices.

        Returns:
          A NamedTuple of Numpy arrays containing the item(s) at the given index(es).
        """
        return self._data._make(value[index] for value in self._data)

    def append(self, item: NamedTuple) -> int:
        """Append a new item to the replay buffer.

        If the buffer is empty, it is allocated to full capacity using Numpy
        arrays of shapes and data types of the provided item.

        Returns:
          The index at which the item was stored.
        """
        # Allocate the buffer on the first append.
        if not self._len:
            self._data = item._make([np.empty((self._max_length, *value.shape), dtype=value.dtype)
                                     for value in map(np.asarray, item)])

        # Select the index to store the new item, updating the sizes.
        if self._len < self._max_length:
            index, self._len = self._len, self._len + 1
        else:
            index, self._offset = self._offset, (self._offset + 1) % self._max_length

        # Store the new item on the given index.
        for i, value in enumerate(item):
            self._data[i][index] = value

        return index

    def append_batch(self, batch: NamedTuple) -> None:
        """Append a named tuple with a batch of items to the replay buffer."""
        if not self._len:
            self._data = batch._make([np.empty((self._max_length, *value.shape[1:]), dtype=value.dtype)
                                      for value in map(np.asarray, batch)])

        batch_size = len(batch[0])
        assert all(len(value) == batch_size for value in batch), "The batch must contain items of the same size"
        assert batch_size > 0, "The batch must contain at least one item"

        if self._len + batch_size <= self._max_length:
            # Append the batch to the end of the buffer in a single call.
            for i, value in enumerate(batch):
                self._data[i][self._len:self._len + batch_size] = value
            self._len += batch_size
        elif self._len == self._max_length and self._offset + batch_size <= self._max_length:
            # Overwrite the oldest items in the buffer in a single call.
            for i, value in enumerate(batch):
                self._data[i][self._offset:self._offset + batch_size] = value
            self._offset = (self._offset + batch_size) % self._max_length
        else:
            # Add batch items one by one to handle wrap-around.
            for batch_index in range(batch_size):
                # Select the index to store the new item, updating the sizes.
                if self._len < self._max_length:
                    index, self._len = self._len, self._len + 1
                else:
                    index, self._offset = self._offset, (self._offset + 1) % self._max_length
                # Store the new item on the given index.
                for i, value in enumerate(batch):
                    self._data[i][index] = value[batch_index]

    def extend(self, items: Sequence[NamedTuple]) -> None:
        """Append a sequence of items to the replay buffer.

        Equivalent to calling append() for each item in the sequence."""
        for item in items:
            self.append(item)

    def sample(self, size: int, replace: bool = True) -> NamedTuple:
        """Sample a batch of items from the replay buffer without replacement.

        Parameters:
          size: The number of items to sample.
          replace: If True, the same item can be sampled multiple times, but the
            random number generator is about twice as fast in this mode.

        Returns:
          A named tuple of Numpy arrays containing the sampled items.
        """
        if replace:
            indices = self._generator.integers(0, self._len, size)
        else:
            indices = self._generator.choice(self._len, size, replace=False)
        return self[indices]
