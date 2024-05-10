#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <utility>

class AZQuiz {
 public:
  static inline const int ACTIONS = 28;
  static inline const int N = 7;
  static inline const int C = 4;

  std::array<int8_t, N * N> board;
  bool randomized;
  int8_t to_play;
  int8_t winner;

  AZQuiz(bool randomized=false) : board(), randomized(randomized), to_play(0), winner(-1) {
    for (int y = 0; y < N; y++)
      for (int x = y + 1; x < N; x++)
        board[y * N + x] = -1;
  }
  AZQuiz(const AZQuiz& other) : board(other.board), randomized(other.randomized), to_play(other.to_play), winner(other.winner) {}

  bool valid(int action) const {
    return winner < 0 && action >= 0 && action < ACTIONS && board[ACTION[action]] < 2;
  }

  void move(int action, bool success=true) {
    if (board[ACTION[action]] == 0) {
      board[ACTION[action]] = success ? 2 + to_play : 1;
    } else {
      if (!success)
        to_play = 1 - to_play;
      board[ACTION[action]] = 2 + to_play;
    }
    to_play = 1 - to_play;

    std::array<bool, N * N> visited{};
    for (int y = 0; y < N; y++) {
      bool edge_right = false, edge_bottom = false;
      int field = board[y * N];
      if (field >= 2) {
        traverse(y, 0, field, visited, edge_right, edge_bottom);
        if (edge_right && edge_bottom) {
          winner = field - 2;
          break;
        }
      }
    }
  }

  std::pair<std::pair<float, AZQuiz>, std::pair<float, AZQuiz>> all_moves(int action) const {
    auto success = *this;
    success.move(action, true);

    auto failure = *this;
    failure.move(action, false);

    float success_probability = board[ACTION[action]] == 0 ? INITIAL_QUESTION_PROB : ADDITIONAL_QUESTION_PROB;
    return {{success_probability, success}, {1 - success_probability, failure}};
  }

  void representation(float* output) const {
    // TODO: This representation function does not currently indicate which player
    // is currently on move, you need to modify it so it does. You can either make
    // the current player have a fixed ID (0 or 1), or you can add another channel
    // indicating the current player (increase `AZQuiz::C` in that case).
    for (auto field : board) {
        *output++ = field == 2;
        *output++ = field == 3;
        *output++ = field == 1;
        *output++ = field >= 0;
    }
  }

 private:
  void traverse(int y, int x, int field, std::array<bool, N * N>& visited, bool& edge_right, bool& edge_bottom) const {
    int pos = y * N + x;
    visited[pos] = true;

    edge_right |= y == x;
    edge_bottom |= y == N - 1;

    if (y - 1 >= 0) {
      if (board[pos - N] == field && !visited[pos - N]) traverse(y - 1, x, field, visited, edge_right, edge_bottom);
      if (x - 1 >= 0 && board[pos - N - 1] == field && !visited[pos - N - 1]) traverse(y - 1, x - 1, field, visited, edge_right, edge_bottom);
    }
    if (x - 1 >= 0 && board[pos - 1] == field && !visited[pos - 1]) traverse(y, x - 1, field, visited, edge_right, edge_bottom);
    if (x + 1 < N && board[pos + 1] == field && !visited[pos + 1]) traverse(y, x + 1, field, visited, edge_right, edge_bottom);
    if (y + 1 < N) {
      if (board[pos + N] == field && !visited[pos + N]) traverse(y + 1, x, field, visited, edge_right, edge_bottom);
      if (x + 1 < N && board[pos + N + 1] == field && !visited[pos + N + 1]) traverse(y + 1, x + 1, field, visited, edge_right, edge_bottom);
    }
  }

  static inline const float INITIAL_QUESTION_PROB = 0.8;
  static inline const float ADDITIONAL_QUESTION_PROB = 0.7;
  static inline const int ACTION[] = {
    0, 7, 8, 14, 15, 16, 21, 22, 23, 24, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48};
};

inline std::ostream& operator<<(std::ostream& os, const AZQuiz& game) {
  if (game.winner < 0)
    os << "Game running, current player: " << "OX"[game.to_play] << std::endl;
  else
    os << "Game finished, winning player: " << "OX"[game.winner] << std::endl;

  for (int y = 0; y < game.N; y++) {
    os.write("       ", game.N - y - 1);
    for (int x = 0; x <= y; x++)
      os << ".*OX"[game.board[y * game.N + x]] << ' ';
    os << std::endl;
  }
  return os;
}
