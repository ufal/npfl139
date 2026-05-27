# This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import re

from .task import Task


class ArithmeticTask(Task):
    """A simple arithmetic task where an expression with 2-5 numbers and +, -, * must be evaluates."""

    @property
    def instructions(self) -> str:
        return " ".join([
            "Solve an arithmetic problem.",
            "Reason step by step and then write the final integer after 'Answer:'.",
        ])

    def create_example(self) -> Task.Example:
        """Create a single example of the arithmetic task.

        The example consists of an expression with 2-5 numbers (0-9) and operators (+, -, *).
        """
        N = self._generator.randint(2, 5)
        numbers = [self._generator.randint(0, 9) for _ in range(N)]
        operators = [self._generator.choice(["+", "-", "*"]) for _ in range(N - 1)]

        stack = [numbers[0]]
        for op, num in zip(operators, numbers[1:]):
            if op == "+":
                stack.append(num)
            elif op == "-":
                stack.append(-num)
            elif op == "*":
                stack[-1] *= num
        result = sum(stack)

        return Task.Example(
            "Compute " + " ".join(f"{num} {op}" for num, op in zip(numbers, operators)) + f" {numbers[-1]}.",
            str(result),
        )

    def extract_answer(self, response: str) -> str | None:
        """Extract the answer from the model's response, if any.

        The answer is expected to be an integer following the case-insensitive pattern "Answer: <integer>",
        optionally followed by a period, at the end of the response.
        """
        if match := re.search(r"Answer:\s*(-?\d+)\s*\.?\s*$", response, re.IGNORECASE):
            return match.group(1)
        return None


# Register the ArithmeticTask.
Task.register_task("arithmetic", ArithmeticTask)
