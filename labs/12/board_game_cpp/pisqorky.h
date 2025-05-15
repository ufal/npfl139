// This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include "board_game.h"

class Pisqorky {
 public:
  static inline const int ACTIONS = 225;
  static inline const int N = 15;
  static inline const int C = 3;

  std::array<int8_t, N * N> board;
  int8_t to_play;
  Outcome current_outcome;

  Outcome outcome(int8_t player) const {
    return !current_outcome || player == to_play ? current_outcome : Outcome(4 - current_outcome);
  }

  Pisqorky() : board(), to_play(0), current_outcome(Outcome::UNFINISHED) {}

  bool valid(int action) const {
    return !current_outcome && action >= 0 && action < ACTIONS && board[action] == 0;
  }

  void move(int action) {
    board[action] = 1 + to_play;
    to_play = 1 - to_play;

    // Check for winner
    bool free_fields = false;
    for (int y = 0; y < N; y++)
      for (int x = 0; x < N; x++) {
        if (board[y * N + x] == 0) {
          free_fields = true;
          continue;
        }
        auto offset = y * N + x;
        auto field = board[offset];
        if ((x >= 4 && y + 4 < N &&
             field == board[offset + N - 1] && field == board[offset + 2 * N - 2] &&
             field == board[offset + 3 * N - 3] && field == board[offset + 4 * N - 4]) ||
            (y + 4 < N &&
             field == board[offset + N] && field == board[offset + 2 * N] &&
             field == board[offset + 3 * N] && field == board[offset + 4 * N]) ||
            (x + 4 < N && y + 4 < N &&
             field == board[offset + N + 1] && field == board[offset + 2 * N + 2] &&
             field == board[offset + 3 * N + 3] && field == board[offset + 4 * N + 4]) ||
            (x + 4 < N &&
             field == board[offset + 1] && field == board[offset + 2] &&
             field == board[offset + 3] && field == board[offset + 4])) {
          current_outcome = field - 1 == to_play ? Outcome::WIN : Outcome::LOSS;
          return;
        }
      }
    if (!free_fields)
      current_outcome = Outcome::DRAW;
  }

  void board_features(float* output) const {
    // TODO: This representation function does not currently indicate which player
    // is currently on move, you need to modify it so it does. You can either make
    // the current player have a fixed ID (0 or 1), or you can add another channel
    // indicating the current player (increase `Pisqorky::C` in that case).
    for (auto field : board) {
        *output++ = field == 0;
        *output++ = field == 1;
        *output++ = field == 2;
    }
  }

 private:
};

inline std::ostream& operator<<(std::ostream& os, const Pisqorky& game) {
  auto outcome = game.outcome(0);
  if (!outcome)
    os << "Game running, current player: " << "OX"[game.to_play] << std::endl;
  else
    os << "Game finished, winning player: " << "X-O"[outcome - 1] << std::endl;

  for (int y = 0; y < game.N; y++) {
    for (int x = 0; x < game.N; x++)
      os << ".OX"[game.board[y * game.N + x]];
    os << std::endl;
  }
  return os;
}
