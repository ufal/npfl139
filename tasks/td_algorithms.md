### Assignment: td_algorithms
#### Date: Deadline: Mar 12, 22:00
#### Points: 4 points
#### Tests: td_algorithms_tests
#### Examples: td_algorithms_examples

Starting with the [td_algorithms.py](https://github.com/ufal/npfl139/tree/master/labs/03/td_algorithms.py)
template, implement all of the following $n$-step TD methods variants:
- SARSA, expected SARSA and Tree backup;
- either on-policy (with $ε$-greedy behaviour policy) or off-policy
  (with the same behaviour policy, but greedy target policy).

#### Tests Start: td_algorithms_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 td_algorithms.py --episodes=10 --mode=sarsa --n=1`
```
Episode 10, mean 100-episode return -652.70 +-37.77
```

2. `python3 td_algorithms.py --episodes=10 --mode=sarsa --n=1 --off_policy`
```
Episode 10, mean 100-episode return -632.90 +-126.41
```

3. `python3 td_algorithms.py --episodes=10 --mode=sarsa --n=4`
```
Episode 10, mean 100-episode return -715.70 +-156.56
```

4. `python3 td_algorithms.py --episodes=10 --mode=sarsa --n=4 --off_policy`
```
Episode 10, mean 100-episode return -649.10 +-171.73
```

5. `python3 td_algorithms.py --episodes=10 --mode=expected_sarsa --n=1`
```
Episode 10, mean 100-episode return -641.90 +-122.11
```

6. `python3 td_algorithms.py --episodes=10 --mode=expected_sarsa --n=1 --off_policy`
```
Episode 10, mean 100-episode return -633.80 +-63.61
```

7. `python3 td_algorithms.py --episodes=10 --mode=expected_sarsa --n=4`
```
Episode 10, mean 100-episode return -713.90 +-107.05
```

8. `python3 td_algorithms.py --episodes=10 --mode=expected_sarsa --n=4 --off_policy`
```
Episode 10, mean 100-episode return -648.20 +-107.08
```

9. `python3 td_algorithms.py --episodes=10 --mode=tree_backup --n=1`
```
Episode 10, mean 100-episode return -641.90 +-122.11
```

10. `python3 td_algorithms.py --episodes=10 --mode=tree_backup --n=1 --off_policy`
```
Episode 10, mean 100-episode return -633.80 +-63.61
```

11. `python3 td_algorithms.py --episodes=10 --mode=tree_backup --n=4`
```
Episode 10, mean 100-episode return -663.50 +-111.78
```

12. `python3 td_algorithms.py --episodes=10 --mode=tree_backup --n=4 --off_policy`
```
Episode 10, mean 100-episode return -708.50 +-125.63
```
#### Tests End:
#### Examples Start: td_algorithms_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 td_algorithms.py --mode=sarsa --n=1`
```
Episode 200, mean 100-episode return -235.23 +-92.94
Episode 400, mean 100-episode return -133.18 +-98.63
Episode 600, mean 100-episode return -74.19 +-70.39
Episode 800, mean 100-episode return -41.84 +-54.53
Episode 1000, mean 100-episode return -31.96 +-52.14
```

- `python3 td_algorithms.py --mode=sarsa --n=1 --off_policy`
```
Episode 200, mean 100-episode return -227.81 +-91.62
Episode 400, mean 100-episode return -131.29 +-90.07
Episode 600, mean 100-episode return -65.35 +-64.78
Episode 800, mean 100-episode return -34.65 +-44.93
Episode 1000, mean 100-episode return -8.70 +-25.74
```

- `python3 td_algorithms.py --mode=sarsa --n=4`
```
Episode 200, mean 100-episode return -277.55 +-146.18
Episode 400, mean 100-episode return -87.11 +-152.12
Episode 600, mean 100-episode return -6.95 +-23.28
Episode 800, mean 100-episode return -1.88 +-19.21
Episode 1000, mean 100-episode return 0.97 +-11.76
```

- `python3 td_algorithms.py --mode=sarsa --n=4 --off_policy`
```
Episode 200, mean 100-episode return -339.11 +-144.40
Episode 400, mean 100-episode return -172.44 +-176.79
Episode 600, mean 100-episode return -36.23 +-100.93
Episode 800, mean 100-episode return -22.43 +-81.29
Episode 1000, mean 100-episode return -3.95 +-17.78
```

- `python3 td_algorithms.py --mode=expected_sarsa --n=1`
```
Episode 200, mean 100-episode return -223.35 +-102.16
Episode 400, mean 100-episode return -143.82 +-96.71
Episode 600, mean 100-episode return -79.92 +-68.88
Episode 800, mean 100-episode return -38.53 +-47.12
Episode 1000, mean 100-episode return -17.41 +-31.26
```

- `python3 td_algorithms.py --mode=expected_sarsa --n=1 --off_policy`
```
Episode 200, mean 100-episode return -231.91 +-87.72
Episode 400, mean 100-episode return -136.19 +-94.16
Episode 600, mean 100-episode return -79.65 +-70.75
Episode 800, mean 100-episode return -35.42 +-44.91
Episode 1000, mean 100-episode return -11.79 +-23.46
```

- `python3 td_algorithms.py --mode=expected_sarsa --n=4`
```
Episode 200, mean 100-episode return -263.10 +-161.97
Episode 400, mean 100-episode return -102.52 +-162.03
Episode 600, mean 100-episode return -7.13 +-24.53
Episode 800, mean 100-episode return -1.69 +-12.21
Episode 1000, mean 100-episode return -1.53 +-11.04
```

- `python3 td_algorithms.py --mode=expected_sarsa --n=4 --off_policy`
```
Episode 200, mean 100-episode return -376.56 +-116.08
Episode 400, mean 100-episode return -292.35 +-166.14
Episode 600, mean 100-episode return -173.83 +-194.11
Episode 800, mean 100-episode return -89.57 +-153.70
Episode 1000, mean 100-episode return -54.60 +-127.73
```

- `python3 td_algorithms.py --mode=tree_backup --n=1`
```
Episode 200, mean 100-episode return -223.35 +-102.16
Episode 400, mean 100-episode return -143.82 +-96.71
Episode 600, mean 100-episode return -79.92 +-68.88
Episode 800, mean 100-episode return -38.53 +-47.12
Episode 1000, mean 100-episode return -17.41 +-31.26
```

- `python3 td_algorithms.py --mode=tree_backup --n=1 --off_policy`
```
Episode 200, mean 100-episode return -231.91 +-87.72
Episode 400, mean 100-episode return -136.19 +-94.16
Episode 600, mean 100-episode return -79.65 +-70.75
Episode 800, mean 100-episode return -35.42 +-44.91
Episode 1000, mean 100-episode return -11.79 +-23.46
```

- `python3 td_algorithms.py --mode=tree_backup --n=4`
```
Episode 200, mean 100-episode return -270.51 +-134.35
Episode 400, mean 100-episode return -64.27 +-109.50
Episode 600, mean 100-episode return -1.80 +-13.34
Episode 800, mean 100-episode return -0.22 +-13.14
Episode 1000, mean 100-episode return 0.60 +-9.37
```

- `python3 td_algorithms.py --mode=tree_backup --n=4 --off_policy`
```
Episode 200, mean 100-episode return -248.56 +-147.74
Episode 400, mean 100-episode return -68.60 +-126.13
Episode 600, mean 100-episode return -6.25 +-32.23
Episode 800, mean 100-episode return -0.53 +-11.82
Episode 1000, mean 100-episode return 2.33 +-8.35
```
#### Examples End:
