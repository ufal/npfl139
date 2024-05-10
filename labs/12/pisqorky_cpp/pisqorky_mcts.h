#pragma once

#include <array>
#include <functional>

#include "pisqorky.h"

typedef std::array<float, Pisqorky::ACTIONS> Policy;

typedef std::function<void(const Pisqorky&, Policy&, float&)> Evaluator;

void mcts(const Pisqorky& game, const Evaluator& evaluator, int num_simulations, float epsilon, float alpha, Policy& policy) {
  // TODO: Implement MCTS, returning the generated `policy`.
  //
  // To run the neural network, use the given `evaluator`, which returns a policy and
  // a value function for the given game.
}
