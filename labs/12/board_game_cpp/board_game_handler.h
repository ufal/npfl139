// This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "az_quiz.h"
#include "pisqorky.h"

#include "mcts.h"
#include "sim_game.h"

namespace py = pybind11;

template <typename T> using np_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

template<BoardGame G>
struct BoardGameHandler {
  SimGame<G> sim_game;

  np_array<float> mcts(np_array<int8_t> board, int8_t to_play,
                       const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& evaluate,
                       int num_simulations, float epsilon, float alpha) {
    if (board.ndim() != 2 || board.shape(0) != G::N || board.shape(1) != G::N)
      throw std::invalid_argument("The BoardGameHandler.mcts got a board of incorrect shape");

    G game;
    std::copy_n(board.data(), game.board.size(), game.board.begin());
    game.to_play = to_play;

    Evaluator<G> evaluator = [&evaluate] (const G& game, Policy<G>& policy, float& value) {
      auto board = np_array<float>({1, game.N, game.N, game.C});
      game.board_features(board.mutable_data());
      auto [np_policy, np_value] = evaluate(board);
      if (np_policy.ndim() != 2 || np_policy.shape(0) != 1 || np_policy.shape(1) != G::ACTIONS)
        throw std::invalid_argument("The evaluator given to BoardGameHandler.mcts returned a policy of incorrect shape");
      if (np_value.ndim() != 1 || np_value.shape(0) != 1)
        throw std::invalid_argument("The evaluator given to BoardGameHandler.mcts returned a value of incorrect shape");
      std::copy_n(np_policy.data(), policy.size(), policy.begin());
      value = *np_value.data();
    };

    Policy<G> policy;
    ::mcts(game, evaluator, num_simulations, epsilon, alpha, policy);

    return np_array<float>(policy.size(), policy.data());
  }

  void simulated_games_start(int threads, int num_simulations, int sampling_moves, float epsilon, float alpha) {
    sim_game.simulated_games_start(threads, num_simulations, sampling_moves, epsilon, alpha);
  }

  py::list simulated_game(const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& evaluate) {
    BatchEvaluator<G> evaluator = [&evaluate] (const Batch<G>& batch) {
      auto board = np_array<float>({int(batch.size()), G::N, G::N, G::C});
      auto board_data = board.mutable_data();
      for (auto& [game, policy, value] : batch) {
        game->board_features(board_data);
        board_data += G::N * G::N * G::C;
      }
      auto [policies, values] = evaluate(board);
      if (policies.ndim() != 2 || policies.shape(0) != int(batch.size()) || policies.shape(1) != G::ACTIONS)
        throw std::invalid_argument("The evaluator given to BoardGameHandler.simulated_game returned a policy of incorrect shape");
      if (values.ndim() != 1 || values.shape(0) != int(batch.size()))
        throw std::invalid_argument("The evaluator given to BoardGameHandler.simulated_game returned a value of incorrect shape");
      auto policies_data = policies.data();
      auto values_data = values.data();
      for (auto& [game, policy, value] : batch) {
        std::copy_n(policies_data, policy->size(), policy->begin());
        policies_data += policy->size();
        *value = *values_data++;
      }
    };

    auto history = sim_game.simulated_game(evaluator);

    py::list results{};
    for (auto& [game, policy, value] : *history) {
      auto board = np_array<float>({game.N, game.N, game.C});
      game.board_features(board.mutable_data());
      results.append(py::make_tuple(board, np_array<float>(policy.size(), policy.data()), value));
    }
    return results;
  }

  void simulated_games_stop() {
    sim_game.simulated_games_stop();
  }
};
