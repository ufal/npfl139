### Assignment: importance_sampling
#### Date: Deadline: Mar 19, 22:00
#### Points: 2 points
#### Tests: importance_sampling_tests

Using the [FrozenLake-v1 environment](https://gymnasium.farama.org/environments/toy_text/frozen_lake/),
implement Monte Carlo weighted importance sampling to estimate
state value function $V$ of target policy, which uniformly chooses either action
1 (down) or action 2 (right), utilizing behavior policy, which uniformly
chooses among all four actions.

Start with the [importance_sampling.py](https://github.com/ufal/npfl139/tree/past-2425/labs/03/importance_sampling.py)
template, which creates the environment and generates episodes according to
behavior policy.

#### Tests Start: importance_sampling_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 importance_sampling.py --episodes=200`
```
 0.00  0.00  0.24  0.32
 0.00  0.00  0.40  0.00
 0.00  0.00  0.20  0.00
 0.00  0.00  0.22  0.00
```

2. `python3 importance_sampling.py --episodes=5000`
```
 0.03  0.00  0.01  0.03
 0.04  0.00  0.09  0.00
 0.10  0.24  0.23  0.00
 0.00  0.44  0.49  0.00
```

3. `python3 importance_sampling.py --episodes=50000`
```
 0.03  0.02  0.05  0.01
 0.13  0.00  0.07  0.00
 0.21  0.33  0.36  0.00
 0.00  0.35  0.76  0.00
```
#### Tests End:
