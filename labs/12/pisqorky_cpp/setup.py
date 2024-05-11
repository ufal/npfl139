from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    name="Pisqorky C++ Module",
    version="0.0.1",
    ext_modules=[Pybind11Extension(
        "pisqorky_cpp", ["pisqorky_cpp.cpp"], depends=[
            "pisqorky.h", "pisqorky_heuristic.h", "pisqorky_mcts.h", "pisqorky_sim_game.h"], cxx_std=17,
    )],
)
