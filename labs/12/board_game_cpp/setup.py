from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    name="C++ Module for board game simulations",
    version="2425.12.0",
    ext_modules=[Pybind11Extension("board_game_cpp", ["board_game_cpp.cpp"], depends=[
        "board_game.h", "board_game_handler.h", "az_quiz.h", "mcts.h", "sim_game.h"], cxx_std=20,
    )],
)
