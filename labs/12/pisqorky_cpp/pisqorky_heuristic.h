#pragma once

#include <cstdint>
#include <random>

#include "pisqorky.h"

int pisqorky_heuristic(const Pisqorky& game) {
  static std::mt19937 generator{std::random_device()()};
  static std::uniform_real_distribution<float> uniform_distribution;

  static const int dir_x[4] = {-1, 0, 1, 1};
  static const int dir_y[4] = {1, 1, 1, 0};
  static const int coefs[2][5] = {{0, 5, 25, 125, 1000}, {0, 4, 20, 100, 400}};

  int best_score = -1;
  int best_move = -1;
  int best_multiplicity = -1;
  bool first_move = true;

  for (int y = 0; y < Pisqorky::N; y++)
    for (int x = 0; x < Pisqorky::N; x++) {
      if (game.board[y * Pisqorky::N + x] != 0) {
        first_move = false;
        continue;
      }

      int score = 0;
      for (int dir = 0; dir < 4; dir++)
        for (int shift = -4; shift <= 0; shift++) {
          int counts[3] = {0, 0, 0};
          for (int stone = 0; stone < 5; stone++) {
            int sx = x + dir_x[dir] * (shift + stone);
            int sy = y + dir_y[dir] * (shift + stone);
            if (sx < 0 || sx >= Pisqorky::N || sy < 0 || sy >= Pisqorky::N) {
              counts[0]++;
            } else {
              auto field = game.board[sy * Pisqorky::N + sx];
              if (field > 0)
                counts[2 - (field - 1 == game.to_play)]++;
            }
          }
          if (!counts[0] && (counts[1] == 0 || counts[2] == 0))
            score += coefs[0][counts[1]] + coefs[1][counts[2]];
        }

      if (score > best_score) {
        best_score = score;
        best_move = y * Pisqorky::N + x;
        best_multiplicity = 1;
      } else if (score == best_score) {
        best_multiplicity++;
        if (uniform_distribution(generator) < 1.0 / best_multiplicity)
          best_move = y * Pisqorky::N + x;
      }
    }

  if (first_move) {
    int y = Pisqorky::N / 2 + int(uniform_distribution(generator) * 3) - 1;
    int x = Pisqorky::N / 2 + int(uniform_distribution(generator) * 3) - 1;
    best_move = y * Pisqorky::N + x;
  }

  return best_move;
}
