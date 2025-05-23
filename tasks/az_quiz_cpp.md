### Assignment: az_quiz_cpp

In addition to the Python template for `az_quiz`, you can also use
[board_game_cpp](https://github.com/ufal/npfl139/tree/master/labs/12/board_game_cpp),
which is a directory containing a skeleton of C++ MCTS and self-play implementation.
Utilizing the C++ implementation is not required, but it offers a large speedup
(up to 10 times on a multi-core CPU and up to 100 times on a GPU). See
[README.md](https://github.com/ufal/npfl139/tree/master/labs/12/board_game_cpp/README.md)
for more information.

In ReCodEx, you can submit the C++ implementation directly in the `az_quiz`
assignment, by including all the C++ headers and sources plus the `setup.py` module
as a part of your submission. Then the `setup.py` model is present, ReCodEx
first compiles your submission using `python3 setup.py build --build-platlib .`
