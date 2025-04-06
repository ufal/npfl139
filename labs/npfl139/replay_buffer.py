# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Generic, Sequence, TypeVar

import numpy as np

Item = TypeVar("Item")  # A generic type for the items in the replay buffer.


class ReplayBuffer(Generic[Item]):
    """Simple replay buffer with possibly limited capacity.

    We use a custom implementation instead of `collections.deque`, which has
    linear complexity of indexing (it is a two-way linked list). The following
    implementation has similar runtime performance as a numpy array of objects,
    but it has unnecessary memory overhead (hundreds of MBs for 1M elements).
    Using five numpy arrays (for state, action, reward, done, and next state)
    would provide minimal memory overhead, but is less flexible.
    """
    def __init__(self, max_length: int | None = None):
        if max_length is not None:
            assert isinstance(max_length, int), "The max_length argument must be an integer"
            assert max_length > 0, "The max_length argument must be a positive integer"
        self._max_length = max_length
        self._data = []
        self._offset = 0

    def __len__(self) -> int:
        return len(self._data)

    @property
    def max_length(self) -> int | None:
        return self._max_length

    def append(self, item: Item) -> None:
        if self._max_length is not None and len(self._data) >= self._max_length:
            self._data[self._offset] = item
            self._offset = (self._offset + 1) % self._max_length
        else:
            self._data.append(item)

    def extend(self, items: Sequence[Item]) -> None:
        if self._max_length is None:
            self._data.extend(items)
        else:
            for item in items:
                if len(self._data) >= self._max_length:
                    self._data[self._offset] = item
                    self._offset = (self._offset + 1) % self._max_length
                else:
                    self._data.append(item)

    def __getitem__(self, index: int) -> Item:
        assert -len(self._data) <= index < len(self._data)
        return self._data[(self._offset + index) % len(self._data)]

    def sample(self, size: int, generator=np.random, replace: bool = True) -> list[Item]:
        # By default, the same element can be sampled multiple times. Making sure the samples
        # are unique is costly, and we do not mind the duplicites much during training.
        if replace:
            return [self._data[index] for index in generator.randint(len(self._data), size=size)]
        else:
            return [self._data[index] for index in generator.choice(len(self._data), size=size, replace=False)]
