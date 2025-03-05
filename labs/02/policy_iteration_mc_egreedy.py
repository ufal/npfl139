#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
parser.add_argument("--epsilon", default=0.02, type=float, help="Monte Carlo epsilon")
parser.add_argument("--mc_length", default=100, type=int, help="Monte Carlo simulation episode length")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")
# If you add more arguments, ReCodEx will keep them with your default values.


class GridWorld:
    # States in the gridworld are the following:
    # 0 1 2 3
    # 4 x 5 6
    # 7 8 9 10

    # The rewards are +1 in state 10 and -100 in state 6

    # Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
    # with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
    states: int = 11
    actions: int = 4
    action_labels: list[str] = ["↑", "→", "↓", "←"]

    def __init__(self, seed: int) -> None:
        self._generator = np.random.RandomState(seed)

    def step(self, state: int, action: int) -> tuple[float, int]:
        probability = self._generator.uniform()
        if probability <= 0.8:
            return self._step(state, action)
        elif probability <= 0.9:
            return self._step(state, (action + 1) % 4)
        else:
            return self._step(state, (action + 3) % 4)

    def epsilon_greedy(self, epsilon: float, greedy_action: int) -> int:
        if self._generator.uniform() < epsilon:
            return self._generator.randint(self.actions)
        return greedy_action

    @staticmethod
    def _step(state: int, action: int) -> tuple[float, int]:
        state += (state >= 5)
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not (new_x >= 4 or new_x < 0 or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        state -= (state >= 5)
        return (+1 if state == 10 else -100 if state == 6 else 0, state)


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def main(args: argparse.Namespace) -> tuple[list[float] | np.ndarray, list[int] | np.ndarray]:
    env = GridWorld(args.seed)

    # Start with zero action-value function and "go North" policy
    action_value_function = np.zeros((env.states, env.actions))
    policy = np.zeros(env.states, np.int32)

    # Note that this assignment is an extension of policy_iteration_mc_estarts,
    # so it is probably easier to implement it first.

    # TODO: Implement a variant of policy iteration algorithm, with
    # `args.steps` steps of policy evaluation/policy improvement. During policy
    # evaluation, estimate action-value function by Monte Carlo simulation:
    # - for start_state in range(env.states):
    #   - start in the given `start_state`
    #   - perform `args.mc_length` Monte Carlo steps, utilizing
    #     epsilon-greedy actions with respect to the policy, using
    #     `env.epsilon_greedy(args.epsilon, policy[current_state])`
    #     - this method returns a random action with probability `args.epsilon`
    #     - otherwise it returns the passed `policy[current_state]` action
    #     - for replicability, make sure to call it exactly `args.mc_length`
    #       times in every simulation
    #   - compute the return of the simulation
    #   - update the action-value function at the `(start_state, start action)`
    #     pair, considering the simulation return as its estimate, by averaging
    #     all estimates from this and previous steps of policy evaluation.
    #
    # After completing the policy_evaluation step (i.e., after updating estimates
    # in all states), perform the policy improvement, using the
    # `argmax_with_tolerance` to choose the best action.

    # TODO: Compute `value_function` by taking the value from
    # `action_value_function` according to the computed policy.
    value_function = None

    return value_function, policy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    value_function, policy = main(main_args)

    # Print results
    for r in range(3):
        for c in range(4):
            state = 4 * r + c
            state -= (state >= 5)
            print("        " if r == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
            print(" " if r == 1 and c == 1 else GridWorld.action_labels[policy[state]], end="")
        print()
