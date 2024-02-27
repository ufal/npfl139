### Assignment: policy_iteration_mc_estarts
#### Date: Deadline: Mar 12, 22:00
#### Points: 2 points
#### Tests: policy_iteration_mc_estarts_tests

Starting with [policy_iteration_mc_estarts.py](https://github.com/ufal/npfl139/tree/master/labs/02/policy_iteration_mc_estarts.py),
extend the `policy_iteration` assignment to perform policy evaluation
by using Monte Carlo estimation with exploring starts. Specifically,
we update the action-value function $q_\pi(s, a)$ by running a
simulation with a given number of steps and using the observed return
as its estimate.

The estimation can now be performed model-free (without the access to the full
MDP dynamics), therefore, the `GridWorld.step` returns a randomly sampled
result instead of a full distribution.

#### Tests Start: policy_iteration_mc_estarts_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 policy_iteration_mc_estarts.py --gamma=0.95 --seed=42 --steps=1`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑             0.00↑    0.00↑
    0.00↑    0.00→    0.00↑    0.00↓
```

2. `python3 policy_iteration_mc_estarts.py --gamma=0.95 --seed=42 --steps=10`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑             0.00↑  -19.50↑
    0.27↓    0.48←    2.21↓    8.52↓
```

3. `python3 policy_iteration_mc_estarts.py --gamma=0.95 --seed=42 --steps=50`
```
    0.09↓    0.32↓    0.22←    0.15↑
    0.18↑            -2.43←   -5.12↓
    0.18↓    1.80↓    3.90↓    9.14↓
```

4. `python3 policy_iteration_mc_estarts.py --gamma=0.95 --seed=42 --steps=100`
```
    3.09↓    2.42←    2.39←    1.17↑
    3.74↓             1.66←    0.18←
    3.92→    5.28→    7.16→   11.07↓
```

5. `python3 policy_iteration_mc_estarts.py --gamma=0.95 --seed=42 --steps=200`
```
    7.71↓    6.76←    6.66←    3.92↑
    8.27↓             6.17←    5.31←
    8.88→   10.12→   11.36→   13.92↓
```
#### Tests End:
