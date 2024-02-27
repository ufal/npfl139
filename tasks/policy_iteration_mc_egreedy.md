### Assignment: policy_iteration_mc_egreedy
#### Date: Deadline: Mar 12, 22:00
#### Points: 2 points
#### Tests: policy_iteration_mc_egreedy_tests

Starting with [policy_iteration_mc_egreedy.py](https://github.com/ufal/npfl139/tree/master/labs/02/policy_iteration_mc_egreedy.py),
extend the `policy_iteration_mc_estarts` assignment to perform policy
evaluation by using $ε$-greedy Monte Carlo estimation. Specifically,
we update the action-value function $q_\pi(s, a)$ by running a
simulation with a given number of steps and using the observed return
as its estimate.

For the sake of replicability, use the provided
`GridWorld.epsilon_greedy(epsilon, greedy_action)` method, which returns
a random action with probability of `epsilon` and otherwise returns the
given `greedy_action`.

#### Tests Start: policy_iteration_mc_egreedy_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 policy_iteration_mc_egreedy.py --gamma=0.95 --seed=42 --steps=1`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑             0.00→    0.00→
    0.00↑    0.00↑    0.00→    0.00→
```

2. `python3 policy_iteration_mc_egreedy.py --gamma=0.95 --seed=42 --steps=10`
```
   -1.20↓   -1.43←    0.00←   -6.00↑
    0.78→           -20.26↓    0.00←
    0.09←    0.00↓   -9.80↓   10.37↓
```

3. `python3 policy_iteration_mc_egreedy.py --gamma=0.95 --seed=42 --steps=50`
```
   -0.16↓   -0.19←    0.56←   -6.30↑
    0.13→            -6.99↓   -3.51↓
    0.01←    0.00←    3.18↓    7.57↓
```

4. `python3 policy_iteration_mc_egreedy.py --gamma=0.95 --seed=42 --steps=100`
```
   -0.07↓   -0.09←    0.28←   -4.66↑
    0.06→            -5.04↓   -8.32↓
    0.00←    0.00←    1.70↓    4.38↓
```

5. `python3 policy_iteration_mc_egreedy.py --gamma=0.95 --seed=42 --steps=200`
```
   -0.04↓   -0.04←   -0.76←   -4.15↑
    0.03→            -8.02↓   -5.96↓
    0.00←    0.00←    2.53↓    4.36↓
```

6. `python3 policy_iteration_mc_egreedy.py --gamma=0.95 --seed=42 --steps=500`
```
   -0.02↓   -0.02←   -0.65←   -3.52↑
    0.01→           -11.34↓   -8.07↓
    0.00←    0.00←    3.15↓    3.99↓
```
#### Tests End:
