from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    name="AZQuiz C++ Module",
    version="0.0.1",
    ext_modules=[Pybind11Extension(
        "az_quiz_cpp", ["az_quiz_cpp.cpp"], depends=["az_quiz.h", "az_quiz_mcts.h", "az_quiz_sim_game.h"], cxx_std=17,
    )],
)
