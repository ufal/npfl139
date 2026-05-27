# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from abc import ABC, abstractmethod
import dataclasses
from typing import TypeAlias

import random


class Task(ABC):
    """Base class for all LLM tasks."""

    @dataclasses.dataclass
    class Example:
        """A single task example."""
        prompt: str
        answer: str

    Dataset: TypeAlias = list[Example]
    """A dataset is a list of examples."""

    def __init__(self, seed: int | None = None) -> None:
        self._generator = random.Random(seed)

    @property
    @abstractmethod
    def instructions(self) -> str:
        """The system instructions for the task, to be given to the model before the examples."""
        ...

    @abstractmethod
    def create_example(self) -> Example:
        """Create a single example of the task."""
        ...

    def create_dataset(self, size: int) -> Dataset:
        """Create a dataset of the given number of examples."""
        return [self.create_example() for _ in range(size)]

    @abstractmethod
    def extract_answer(self, response: str) -> str | None:
        """Extract the answer from the model's response, if any."""
        ...

    def evaluate(self, responses: list[str], dataset: Dataset) -> float:
        """Evaluate the model's responses on the given dataset."""
        assert len(responses) == len(dataset), "Number of responses must match the dataset size."

        correct = 0
        for response, gold in zip(responses, dataset):
            answer = self.extract_answer(response)
            correct += answer is not None and answer == gold.answer
        return correct / len(dataset)

    # Static factory methods
    @staticmethod
    def from_name(task_name: str) -> type["Task"]:
        """Return a task class from the task name."""
        assert task_name in Task._tasks, f"Task.from_name got unknown task name: {task_name}"
        return Task._tasks[task_name]

    @staticmethod
    def register_task(task_name: str, task_class: type["Task"]) -> None:
        """Register a new task class."""
        Task._tasks[task_name] = task_class

    _tasks: dict[str, type["Task"]] = {}
