// This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <concepts>
#include <array>
#include <cstdint>
#include <random>

enum Outcome: int8_t {
  UNFINISHED = 0,
  LOSS = 1,
  DRAW = 2,
  WIN = 3,
};

template<typename T>
concept BoardGame = requires(T game, int8_t player, int action, float* output) {
  { T::ACTIONS } -> std::convertible_to<int>;
  { T::N } -> std::convertible_to<int>;
  { T::C } -> std::convertible_to<int>;

  { game.board } -> std::convertible_to<std::array<int8_t, T::N * T::N>>;
  { game.board_features(output) } -> std::same_as<void>;
  { game.to_play } -> std::convertible_to<int8_t>;
  { game.outcome(player) } -> std::same_as<Outcome>;
  { game.valid(action) } -> std::same_as<bool>;
  { game.move(action) } -> std::same_as<void>;
};

inline thread_local std::mt19937* board_game_generator = new std::mt19937{std::random_device()()};
