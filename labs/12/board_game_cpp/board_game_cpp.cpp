// This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#include <variant>

#include "board_game_handler.h"

class BoardGameCpp {
  public:
    static void select_game(std::string game_name) {
      if (game_selected)
        throw std::runtime_error("Game already selected");

      if (game_name == "az_quiz") {
        game_handler = new BoardGameHandler<AZQuiz>();
      } else if (game_name == "az_quiz_randomized") {
        game_handler = new BoardGameHandler<AZQuizRandomized>();
      } else {
        throw std::invalid_argument("Unknown game name");
      }
      game_selected = true;
    }

    static np_array<float> mcts(np_array<int8_t> board, int8_t to_play,
                                const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& evaluate,
                                int num_simulations, float epsilon, float alpha) {
      check_game_selected();
      return std::visit([&](auto&& game_h) {
        return game_h->mcts(board, to_play, evaluate, num_simulations, epsilon, alpha);
      }, game_handler);
    }

    static void simulated_games_start(int threads, int num_simulations, int sampling_moves, float epsilon, float alpha) {
      check_game_selected();
      std::visit([&](auto&& game_h) {
        game_h->simulated_games_start(threads, num_simulations, sampling_moves, epsilon, alpha);
      }, game_handler);
    }

    static py::list simulated_game(const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& evaluate) {
      check_game_selected();
      return std::visit( [&](auto&& game_h) {
        return game_h->simulated_game(evaluate);
      }, game_handler);
    }

    static void simulated_games_stop() {
      check_game_selected();
      std::visit([&](auto&& game_h) {
        game_h->simulated_games_stop();
      }, game_handler);
    }
  private:
    static inline std::variant<BoardGameHandler<AZQuiz>*, BoardGameHandler<AZQuizRandomized>*> game_handler;
    static inline bool game_selected = false;

    static void check_game_selected() {
      if (!game_selected) {
        throw std::runtime_error("Game not selected");
      }
    }
};

PYBIND11_MODULE(board_game_cpp, m) {
    m.doc() = "C++ Module for board game simulations";

    m.def("select_game", &BoardGameCpp::select_game, "Select the game to play", py::arg("game_name"));
    m.def("mcts", &BoardGameCpp::mcts, "Run a Monte Carlo Tree Search",
          py::arg("board"), py::arg("to_play"), py::arg("evaluate"), py::arg("num_simulations"), py::arg("epsilon"), py::arg("alpha"));
    m.def("simulated_games_start", &BoardGameCpp::simulated_games_start, "Start generating simulated games",
          py::arg("threads"), py::arg("num_simulations"), py::arg("sampling_moves"), py::arg("epsilon"), py::arg("alpha"));
    m.def("simulated_game", &BoardGameCpp::simulated_game, "Get one simulated game", py::arg("evaluate"));
    m.def("simulated_games_stop", &BoardGameCpp::simulated_games_stop, "Shut down the application including the worker threads");
}
