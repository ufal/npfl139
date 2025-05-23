# Skeleton of C++ BoardGame MCTS Implementation

This directory contains a skeleton of C++ implementation of BoardGame
MCTS and self-play game simulation. The implementation is fully generic
regarding the board game type.

## Prerequisities

You need to
```
pip3 install [--user] pybind11
```
in order to be able to compile the project.

## Compilation

To compile the project, you can run for example
```
python3 setup.py build --build-platlib .
```
which creates the binary `board_game_cpp` module in the current directory.

You can also use `make` to run this command (and `make clean` to remove the
compiled module).

## ReCodEx

The C++ implementation can be used in ReCodEx -- just submit also all the
C++ headers and sources, plus the `setup.py` module. When `setup.py` module
is submitted to ReCodEx, the above compilation command is automatically
run before importing your module; any compilation error should be reported.

## Performance

The notable property of the C++ implementation is that it can run self-play
simulation in several threads in parallel, batching evaluation requests from
all the threads. This allows large speedup both in CPU-only and GPU setups,
as indicated by the below table measurning approximate running times of
generating 1000 self-play games and performing 1000 training updates with
batch size of 512.

| Implementation | Device | Workers (parallel MCTSes) | Time |
|:---------------|:-------|--------------------------:|-----:|
| Python | 1 CPU  |   – | 2359.2s |
| C++    | 1 CPU  |   1 | 1190.3s |
| C++    | 1 CPU  |   4 |  613.3s |
| C++    | 1 CPU  |  16 |  483.1s |
| C++    | 1 CPU  |  64 |  403.8s |
| C++    | 4 CPUs |   1 |  912.8s |
| C++    | 4 CPUs |   4 |  408.2s |
| C++    | 4 CPUs |  16 |  204.7s |
| C++    | 4 CPUs |  64 |  166.2s |
| C++    | GPU    |  64 |   42.4s |
| C++    | GPU    | 128 |   29.7s |
| C++    | GPU    | 256 |   24.8s |
| C++    | GPU    | 512 |   21.3s |

## API Documentation

You can have a look at [az_quiz_cpp_example.py](az_quiz_cpp_example.py) to see
how `board_game_cpp` can be used from Python.

The provided implementation uses C++-20 and contains:
- `board_game.h`, which is a concept definition of a `BoardGame`;
- `az_quiz.h`, which is a C++ reimplementation of `az_quiz.py` fulfilling the
  `BoardGame` concept;
- `board_game_cpp.cpp` and `board_game_handler.h`, which are the implementation
  of the Python `board_game_cpp` module. This module provides the following methods:
  - ```python
    select_game(game_name: string) -> None
    ```
    Select the specified game type (the same name as passed to
    `BoardGame.from_name`).
  - ```python
    mcts(
        board: np.ndarray,
        to_play: int,
        evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        num_simulations: int,
        epsilon: float,
        alpha: float,
    ) -> np.ndarray
    ```
    Run a MCTS and returns the policy computed from the visit counts of the
    root children. The `evaluate` is a callable, which given a batch of game
    representations produces a batch of policies and batch of value functions.
  - ```python
    simulated_games_start(
        threads: int,
        num_simulations: int,
        sampling_moves: int,
        epsilon: float,
        alpha: float,
    ) -> None
    ```
    Start the given number of threads, each performing one self-play game
    simulation. Should be called once at the beginning of the training.
  - ```python
    simulated_game(
        evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> list[tuple[np.ndarray, np.ndarray, float]]
    ```
    Given a callable `evaluate`, run parallel MCTS self-play simulations (using
    threads created by the `simulated_games_start` call). Once the first game
    finishes, it is returned as a list of triples _(game representation, policy,
    value function)_.
  - ```python
    simulated_games_stop() -> None
    ```
    Stop the threads generating the self-play games. Should be called
    once after the training has finished.
- `mcts.h`, where you should implement the MCTS;
- `sim_game.h`, where you should implement the self-play simulation.

The implementation contains all Python ↔ C++ conversions and thread synchronization.
You need to implement:
- `AZQuiz::board_features` to suitably represent a given game;
- `mcts` implementing the MCTS;
- `SimGame::worker_thread` generating a simulation of a self-play game.

Note that all other functionality is assumed to be provided by the Python
implementation (network construction, GPU utilization, training cycle, evaluation, …).
