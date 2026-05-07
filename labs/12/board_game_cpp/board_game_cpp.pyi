"""C++ Module for board game simulations"""

import numpy as np
import numpy.typing as npt
from typing import Callable

def select_game(game_name: str) -> None:
    """Select the game to play"""
    ...

def mcts(
    board: npt.NDArray[np.int8],
    to_play: int,
    evaluate: Callable[
        [npt.NDArray[np.int8]], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ],
    num_simulations: int,
    epsilon: float,
    alpha: float,
) -> npt.NDArray[np.floating]:
    """Run a Monte Carlo Tree Search"""
    ...

def simulated_games_start(
    threads: int,
    num_simulations: int,
    sampling_moves: int,
    epsilon: float,
    alpha: float,
) -> None:
    """Start generating simulated games"""
    ...

def simulated_game(
    evaluate: Callable[
        [npt.NDArray[np.int8]], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ],
) -> list[tuple[npt.NDArray[np.int8], npt.NDArray[np.floating], float]]:
    """Get one simulated game"""
    ...

def simulated_games_stop() -> None:
    """Shut down the application including the worker threads"""
    ...
