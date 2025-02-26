### Assignment: policy_iteration
#### Date: Deadline: Mar 12, 22:00
#### Points: 2 points
#### Tests: policy_iteration_tests

Consider the following gridworld:

![Gridworld example](//ufal.mff.cuni.cz/~straka/courses/npfl139/2425/tasks/figures/policy_iteration.svgz)

Start with [policy_iteration.py](https://github.com/ufal/npfl139/tree/master/labs/02/policy_iteration.py),
which implements the gridworld mechanics, by providing the following methods:
- `GridWorld.states`: return the number of states (`11`)
- `GridWorld.actions`: return a list with labels of the actions (`["↑", "→", "↓", "←"]`)
- `GridWorld.step(state, action)`: return possible outcomes of performing the
  `action` in a given `state`, as a list of triples containing
  - `probability`: probability of the outcome
  - `reward`: reward of the outcome
  - `new_state`: new state of the outcome

Implement policy iteration algorithm, with `--steps` steps of policy
evaluation/policy improvement. During policy evaluation, use the current value
function and perform `--iterations` applications of the Bellman equation.
Perform the policy evaluation asynchronously (i.e., update the value function
in-place for states $0, 1, …$). Assume the initial policy is “go North” and
initial value function is zero.

#### Tests Start: policy_iteration_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 policy_iteration.py --gamma=0.95 --iterations=1 --steps=1`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑           -10.00←  -10.95↑
    0.00↑    0.00←   -7.50←  -88.93←
```

2. `python3 policy_iteration.py --gamma=0.95 --iterations=1 --steps=2`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑            -8.31←  -11.83←
    0.00↑    0.00←   -1.50←  -20.61←
```

3. `python3 policy_iteration.py --gamma=0.95 --iterations=1 --steps=3`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑            -6.46←   -6.77←
    0.00↑    0.00←   -0.76←  -13.08↓
```

4. `python3 policy_iteration.py --gamma=0.95 --iterations=1 --steps=10`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑            -1.04←   -0.83←
    0.00↑    0.00←   -0.11→   -0.34↓
```

5. `python3 policy_iteration.py --gamma=0.95 --iterations=10 --steps=10`
```
   11.93↓   11.19←   10.47←    6.71↑
   12.83↓            10.30←   10.12←
   13.70→   14.73→   15.72→   16.40↓
```

6. `python3 policy_iteration.py --gamma=1 --iterations=1 --steps=100`
```
   74.73↓   74.50←   74.09←   65.95↑
   75.89↓            72.63←   72.72←
   77.02→   78.18→   79.31→   80.16↓
```
#### Tests End:
