#pragma once

#include <array>
#include <cstdint>
#include <iostream>

class Pisqorky {
 public:
  static inline const int ACTIONS = 225;
  static inline const int N = 15;
  static inline const int C = 3;

  std::array<int8_t, N * N> board;
  int8_t to_play;
  int8_t winner;

  Pisqorky() : board(), to_play(0), winner(-1) {}
  Pisqorky(const Pisqorky& other) : board(other.board), to_play(other.to_play), winner(other.winner) {}

  bool valid(int action) const {
    return winner < 0 && action >= 0 && action < ACTIONS && board[action] == 0;
  }

  void move(int action) {
    board[action] = 1 + to_play;
    to_play = 1 - to_play;

    // Check for winner
    for (int y = 0; y < N; y++)
      for (int x = 0; x < N; x++) {
        if (board[y * N + x] == 0)
          continue;
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
          winner = field - 1;
          return;
        }
      }
  }

  void representation(float* output) const {
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
  if (game.winner < 0)
    os << "Game running, current player: " << "OX"[game.to_play] << std::endl;
  else
    os << "Game finished, winning player: " << "OX"[game.winner] << std::endl;

  for (int y = 0; y < game.N; y++) {
    for (int x = 0; x <= y; x++)
      os << ".*OX"[game.board[y * game.N + x]];
    os << std::endl;
  }
  return os;
}
