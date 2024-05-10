#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pisqorky.h"
#include "pisqorky_heuristic.h"
#include "pisqorky_mcts.h"
#include "pisqorky_sim_game.h"

namespace py = pybind11;

class PisqorkyCpp {
 public:
  template <typename T> using np_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

  static int32_t heuristic(const np_array<int8_t>& board, int8_t to_play) {
    Pisqorky game;
    std::copy_n(board.data(), game.board.size(), game.board.begin());
    game.to_play = to_play;
    return pisqorky_heuristic(game);
  }

  static np_array<float> mcts(np_array<int8_t> board, int8_t to_play,
                              const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& network,
                              int num_simulations, float epsilon, float alpha) {
    Pisqorky game;
    std::copy_n(board.data(), game.board.size(), game.board.begin());
    game.to_play = to_play;

    Evaluator evaluator = [&network] (const Pisqorky& game, Policy& policy, float& value) {
      auto board = np_array<float>({1, game.N, game.N, game.C});
      game.representation(board.mutable_data());
      auto [np_policy, np_value] = network(board);
      std::copy_n(np_policy.data(), policy.size(), policy.begin());
      value = *np_value.data();
    };

    Policy policy;
    ::mcts(game, evaluator, num_simulations, epsilon, alpha, policy);

    return np_array<float>(policy.size(), policy.data());
  }

  static void simulated_games_start(int threads, int num_simulations, int sampling_moves, float epsilon, float alpha) {
    ::simulated_games_start(threads, num_simulations, sampling_moves, epsilon, alpha);
  }

  static py::list simulated_game(const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& network) {
    BatchEvaluator evaluator = [&network] (const Batch& batch) {
      auto board = np_array<float>({int(batch.size()), Pisqorky::N, Pisqorky::N, Pisqorky::C});
      auto board_data = board.mutable_data();
      for (auto& [game, policy, value] : batch) {
        game->representation(board_data);
        board_data += Pisqorky::N * Pisqorky::N * Pisqorky::C;
      }
      auto [policies, values] = network(board);
      auto policies_data = policies.data();
      auto values_data = values.data();
      for (auto& [game, policy, value] : batch) {
        std::copy_n(policies_data, policy->size(), policy->begin());
        policies_data += policy->size();
        *value = *values_data++;
      }
    };

    auto history = ::simulated_game(evaluator);

    py::list results{};
    for (auto& [game, policy, value] : *history) {
      auto board = np_array<float>({game.N, game.N, game.C});
      game.representation(board.mutable_data());
      results.append(py::make_tuple(board, np_array<float>(policy.size(), policy.data()), value));
    }
    return results;
  }

  static void simulated_games_stop() {
    ::simulated_games_stop();
  }
};


PYBIND11_MODULE(pisqorky_cpp, m) {
    m.doc() = "Pisqorky C++ Module";

    m.def("heuristic", &PisqorkyCpp::heuristic, "Return an action chosen by a heuristic");
    m.def("mcts", &PisqorkyCpp::mcts, "Run a Monte Carlo Tree Search");
    m.def("simulated_games_start", &PisqorkyCpp::simulated_games_start, "Start generating simulated games");
    m.def("simulated_game", &PisqorkyCpp::simulated_game, "Get one simulated game");
    m.def("simulated_games_stop", &PisqorkyCpp::simulated_games_stop, "Shut down the application including the worker threads");
}
