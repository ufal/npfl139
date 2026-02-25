### Assignment: policy_iteration_exact
#### Date: Deadline: Mar 11, 22:00
#### Points: 2 points
#### Tests: policy_iteration_exact_tests

Starting with [policy_iteration_exact.py](https://github.com/ufal/npfl139/tree/master/labs/02/policy_iteration_exact.py),
extend the `policy_iteration` assignment to perform policy evaluation
exactly by solving a system of linear equations. Note that you need to
use 64-bit floats because lower precision results in unacceptable error.

#### Tests Start: policy_iteration_exact_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._

1. `python3 policy_iteration_exact.py --gamma=0.95 --steps=1`
```
   -0.00↑   -0.00↑   -0.00↑   -0.00↑
   -0.00↑           -12.35←  -12.35↑
   -0.85←   -8.10←  -19.62← -100.71←
```

2. `python3 policy_iteration_exact.py --gamma=0.95 --steps=2`
```
    0.00↑    0.00↑    0.00↑    0.00↑
    0.00↑             0.00←  -11.05←
   -0.00↑   -0.00↑   -0.00←  -12.10↓
```

3. `python3 policy_iteration_exact.py --gamma=0.95 --steps=3`
```
   -0.00↑    0.00↑    0.00↑    0.00↑
   -0.00↑            -0.00←    0.69←
   -0.00↑   -0.00↑   -0.00→    6.21↓
```

4. `python3 policy_iteration_exact.py --gamma=0.95 --steps=4`
```
   -0.00↑    0.00↑    0.00↓    0.00↑
   -0.00↓             5.91←    6.11←
    0.65→    6.17→   14.93→   15.99↓
```

5. `python3 policy_iteration_exact.py --gamma=0.95 --steps=5`
```
    2.83↓    4.32→    8.09↓    5.30↑
   12.92↓             9.44←    9.35←
   13.77→   14.78→   15.76→   16.53↓
```

6. `python3 policy_iteration_exact.py --gamma=0.95 --steps=6`
```
   11.75↓    8.15←    8.69↓    5.69↑
   12.97↓             9.70←    9.59←
   13.82→   14.84→   15.82→   16.57↓
```

7. `python3 policy_iteration_exact.py --gamma=0.95 --steps=7`
```
   12.12↓   11.37←    9.19←    6.02↑
   13.01↓             9.92←    9.79←
   13.87→   14.89→   15.87→   16.60↓
```

8. `python3 policy_iteration_exact.py --gamma=0.95 --steps=8`
```
   12.24↓   11.49←   10.76←    7.05↑
   13.14↓            10.60←   10.42←
   14.01→   15.04→   16.03→   16.71↓
```

9. `python3 policy_iteration_exact.py --gamma=0.9999 --steps=5`
```
 7385.23↓ 7392.62→ 7407.40↓ 7400.00↑
 7421.37↓          7411.10← 7413.16↓
 7422.30→ 7423.34→ 7424.27→ 7425.84↓
```
#### Tests End:
