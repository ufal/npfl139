### Assignment: trace_algorithms
#### Date: Deadline: May 7, 22:00
#### Points: 4 points
#### Tests: trace_algorithms_tests
#### Examples: trace_algorithms_examples

Starting with the [trace_algorithms.py](https://github.com/ufal/npfl139/tree/master/labs/09/trace_algorithms.py)
template, implement the following state value estimations:
- use $n$-step estimates for a given $n$;
- if requested, use truncated lambda return with a given $λ$;
- allow off-policy correction using importance sampling with control variates,
  optionally clipping the individual importance sampling ratios by a given
  threshold.

#### Tests Start: trace_algorithms_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 trace_algorithms.py --episodes=50 --n=1`
```
The mean 1000-episode return after evaluation -196.80 +-25.96
```

2. `python3 trace_algorithms.py --episodes=50 --n=4`
```
The mean 1000-episode return after evaluation -165.45 +-78.01
```

3. `python3 trace_algorithms.py --episodes=50 --n=8 --seed=62`
```
The mean 1000-episode return after evaluation -180.20 +-61.48
```

4. `python3 trace_algorithms.py --episodes=50 --n=4 --trace_lambda=0.6`
```
The mean 1000-episode return after evaluation -170.70 +-72.93
```

5. `python3 trace_algorithms.py --episodes=50 --n=8 --trace_lambda=0.6 --seed=77`
```
The mean 1000-episode return after evaluation -154.24 +-86.67
```

6. `python3 trace_algorithms.py --episodes=50 --n=1 --off_policy`
```
The mean 1000-episode return after evaluation -189.16 +-46.74
```

7. `python3 trace_algorithms.py --episodes=50 --n=4 --off_policy`
```
The mean 1000-episode return after evaluation -159.09 +-83.40
```

8. `python3 trace_algorithms.py --episodes=50 --n=8 --off_policy`
```
The mean 1000-episode return after evaluation -166.82 +-76.04
```

9. `python3 trace_algorithms.py --episodes=50 --n=1 --off_policy --vtrace_clip=1`
```
The mean 1000-episode return after evaluation -198.50 +-17.93
```

10. `python3 trace_algorithms.py --episodes=50 --n=4 --off_policy --vtrace_clip=1`
```
The mean 1000-episode return after evaluation -144.76 +-92.48
```

11. `python3 trace_algorithms.py --episodes=50 --n=8 --off_policy --vtrace_clip=1`
```
The mean 1000-episode return after evaluation -167.63 +-75.87
```

12. `python3 trace_algorithms.py --episodes=50 --n=4 --off_policy --vtrace_clip=1 --trace_lambda=0.6`
```
The mean 1000-episode return after evaluation -186.28 +-52.05
```

13. `python3 trace_algorithms.py --episodes=50 --n=8 --off_policy --vtrace_clip=1 --trace_lambda=0.6`
```
The mean 1000-episode return after evaluation -185.67 +-53.04
```
#### Tests End:
#### Examples Start: trace_algorithms_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

- `python3 trace_algorithms.py --n=1`
```
Episode 100, mean 100-episode return -96.50 +-92.02
Episode 200, mean 100-episode return -53.64 +-76.70
Episode 300, mean 100-episode return -29.03 +-54.84
Episode 400, mean 100-episode return -8.78 +-21.69
Episode 500, mean 100-episode return -14.24 +-41.76
Episode 600, mean 100-episode return -4.57 +-17.56
Episode 700, mean 100-episode return -7.90 +-27.92
Episode 800, mean 100-episode return -2.17 +-16.67
Episode 900, mean 100-episode return -2.07 +-14.01
Episode 1000, mean 100-episode return 0.13 +-13.93
The mean 1000-episode return after evaluation -35.05 +-84.82
```

- `python3 trace_algorithms.py --n=4`
```
Episode 100, mean 100-episode return -74.01 +-89.62
Episode 200, mean 100-episode return -4.84 +-20.95
Episode 300, mean 100-episode return 0.37 +-11.81
Episode 400, mean 100-episode return 1.82 +-8.04
Episode 500, mean 100-episode return 1.28 +-8.66
Episode 600, mean 100-episode return 3.13 +-7.02
Episode 700, mean 100-episode return 0.76 +-8.05
Episode 800, mean 100-episode return 2.05 +-8.11
Episode 900, mean 100-episode return 0.98 +-9.22
Episode 1000, mean 100-episode return 0.29 +-9.13
The mean 1000-episode return after evaluation -11.49 +-60.05
```

- `python3 trace_algorithms.py --n=8 --seed=62`
```
Episode 100, mean 100-episode return -102.83 +-104.71
Episode 200, mean 100-episode return -5.02 +-23.36
Episode 300, mean 100-episode return -0.43 +-13.33
Episode 400, mean 100-episode return 1.99 +-8.89
Episode 500, mean 100-episode return -2.17 +-16.57
Episode 600, mean 100-episode return -2.62 +-19.87
Episode 700, mean 100-episode return 1.66 +-7.81
Episode 800, mean 100-episode return -7.40 +-36.75
Episode 900, mean 100-episode return -5.95 +-34.04
Episode 1000, mean 100-episode return 3.51 +-7.88
The mean 1000-episode return after evaluation 6.88 +-14.89
```

- `python3 trace_algorithms.py --n=4 --trace_lambda=0.6`
```
Episode 100, mean 100-episode return -85.33 +-91.17
Episode 200, mean 100-episode return -16.06 +-39.97
Episode 300, mean 100-episode return -2.74 +-15.78
Episode 400, mean 100-episode return -0.33 +-9.93
Episode 500, mean 100-episode return 1.39 +-9.48
Episode 600, mean 100-episode return 1.59 +-9.26
Episode 700, mean 100-episode return 3.66 +-6.99
Episode 800, mean 100-episode return 2.08 +-7.26
Episode 900, mean 100-episode return 1.32 +-8.76
Episode 1000, mean 100-episode return 3.33 +-7.27
The mean 1000-episode return after evaluation 7.93 +-2.63
```

- `python3 trace_algorithms.py --n=8 --trace_lambda=0.6 --seed=77`
```
Episode 100, mean 100-episode return -118.76 +-105.12
Episode 200, mean 100-episode return -21.82 +-49.91
Episode 300, mean 100-episode return -0.59 +-11.21
Episode 400, mean 100-episode return 2.27 +-8.29
Episode 500, mean 100-episode return 1.65 +-8.52
Episode 600, mean 100-episode return 1.16 +-10.32
Episode 700, mean 100-episode return 1.18 +-9.62
Episode 800, mean 100-episode return 3.35 +-7.34
Episode 900, mean 100-episode return 1.66 +-8.67
Episode 1000, mean 100-episode return 0.86 +-8.56
The mean 1000-episode return after evaluation -11.93 +-60.63
```

- `python3 trace_algorithms.py --n=1 --off_policy`
```
Episode 100, mean 100-episode return -68.47 +-73.52
Episode 200, mean 100-episode return -29.11 +-34.15
Episode 300, mean 100-episode return -20.30 +-31.24
Episode 400, mean 100-episode return -13.44 +-25.04
Episode 500, mean 100-episode return -4.72 +-13.75
Episode 600, mean 100-episode return -3.07 +-17.63
Episode 700, mean 100-episode return -2.70 +-13.81
Episode 800, mean 100-episode return 1.32 +-11.79
Episode 900, mean 100-episode return 0.78 +-8.95
Episode 1000, mean 100-episode return 1.15 +-9.27
The mean 1000-episode return after evaluation -12.63 +-62.51
```

- `python3 trace_algorithms.py --n=4 --off_policy`
```
Episode 100, mean 100-episode return -96.25 +-105.93
Episode 200, mean 100-episode return -26.21 +-74.65
Episode 300, mean 100-episode return -4.84 +-31.78
Episode 400, mean 100-episode return -0.34 +-9.46
Episode 500, mean 100-episode return 1.15 +-8.49
Episode 600, mean 100-episode return 2.95 +-7.20
Episode 700, mean 100-episode return 0.94 +-10.19
Episode 800, mean 100-episode return 0.13 +-9.27
Episode 900, mean 100-episode return 1.95 +-9.69
Episode 1000, mean 100-episode return 1.91 +-7.59
The mean 1000-episode return after evaluation 6.79 +-3.68
```

- `python3 trace_algorithms.py --n=8 --off_policy`
```
Episode 100, mean 100-episode return -180.08 +-112.11
Episode 200, mean 100-episode return -125.56 +-124.82
Episode 300, mean 100-episode return -113.66 +-125.12
Episode 400, mean 100-episode return -77.98 +-117.08
Episode 500, mean 100-episode return -23.71 +-69.71
Episode 600, mean 100-episode return -21.44 +-67.38
Episode 700, mean 100-episode return -2.43 +-16.31
Episode 800, mean 100-episode return 2.38 +-7.42
Episode 900, mean 100-episode return 1.29 +-7.78
Episode 1000, mean 100-episode return 0.84 +-8.37
The mean 1000-episode return after evaluation 7.03 +-2.37
```

- `python3 trace_algorithms.py --n=1 --off_policy --vtrace_clip=1`
```
Episode 100, mean 100-episode return -71.85 +-75.59
Episode 200, mean 100-episode return -29.60 +-39.91
Episode 300, mean 100-episode return -23.11 +-33.97
Episode 400, mean 100-episode return -12.00 +-21.72
Episode 500, mean 100-episode return -5.93 +-15.92
Episode 600, mean 100-episode return -7.69 +-16.03
Episode 700, mean 100-episode return -2.95 +-13.75
Episode 800, mean 100-episode return 0.45 +-9.76
Episode 900, mean 100-episode return 0.65 +-9.36
Episode 1000, mean 100-episode return -1.56 +-11.53
The mean 1000-episode return after evaluation -24.25 +-75.88
```

- `python3 trace_algorithms.py --n=4 --off_policy --vtrace_clip=1`
```
Episode 100, mean 100-episode return -76.39 +-83.74
Episode 200, mean 100-episode return -3.32 +-13.97
Episode 300, mean 100-episode return -0.33 +-9.49
Episode 400, mean 100-episode return 2.20 +-7.80
Episode 500, mean 100-episode return 1.49 +-7.72
Episode 600, mean 100-episode return 2.27 +-8.67
Episode 700, mean 100-episode return 1.07 +-9.07
Episode 800, mean 100-episode return 3.17 +-6.27
Episode 900, mean 100-episode return 3.25 +-7.39
Episode 1000, mean 100-episode return 0.70 +-8.61
The mean 1000-episode return after evaluation 7.70 +-2.52
```

- `python3 trace_algorithms.py --n=8 --off_policy --vtrace_clip=1`
```
Episode 100, mean 100-episode return -110.07 +-106.29
Episode 200, mean 100-episode return -7.22 +-32.31
Episode 300, mean 100-episode return 0.54 +-9.65
Episode 400, mean 100-episode return 2.03 +-7.82
Episode 500, mean 100-episode return 1.64 +-8.63
Episode 600, mean 100-episode return 1.54 +-7.28
Episode 700, mean 100-episode return 2.80 +-7.86
Episode 800, mean 100-episode return 1.69 +-7.26
Episode 900, mean 100-episode return 1.17 +-8.59
Episode 1000, mean 100-episode return 2.39 +-7.59
The mean 1000-episode return after evaluation 7.57 +-2.35
```

- `python3 trace_algorithms.py --n=4 --off_policy --vtrace_clip=1 --trace_lambda=0.6`
```
Episode 100, mean 100-episode return -81.87 +-87.96
Episode 200, mean 100-episode return -15.94 +-29.31
Episode 300, mean 100-episode return -5.24 +-20.41
Episode 400, mean 100-episode return -1.01 +-12.52
Episode 500, mean 100-episode return 1.09 +-9.55
Episode 600, mean 100-episode return 0.73 +-9.15
Episode 700, mean 100-episode return 3.09 +-7.59
Episode 800, mean 100-episode return 3.13 +-7.60
Episode 900, mean 100-episode return 1.30 +-8.72
Episode 1000, mean 100-episode return 3.77 +-7.11
The mean 1000-episode return after evaluation 6.46 +-17.53
```

- `python3 trace_algorithms.py --n=8 --off_policy --vtrace_clip=1 --trace_lambda=0.6`
```
Episode 100, mean 100-episode return -127.86 +-106.40
Episode 200, mean 100-episode return -27.64 +-48.34
Episode 300, mean 100-episode return -12.75 +-35.05
Episode 400, mean 100-episode return -0.38 +-14.28
Episode 500, mean 100-episode return 1.35 +-9.10
Episode 600, mean 100-episode return 0.43 +-10.53
Episode 700, mean 100-episode return 3.11 +-9.26
Episode 800, mean 100-episode return 3.58 +-6.81
Episode 900, mean 100-episode return 1.24 +-8.24
Episode 1000, mean 100-episode return 1.58 +-7.15
The mean 1000-episode return after evaluation 7.93 +-2.67
```
#### Examples End:
