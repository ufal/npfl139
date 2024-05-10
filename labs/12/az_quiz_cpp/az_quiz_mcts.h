#pragma once

#include <array>
#include <functional>

#include "az_quiz.h"

typedef std::array<float, AZQuiz::ACTIONS> Policy;

typedef std::function<void(const AZQuiz&, Policy&, float&)> Evaluator;

void mcts(const AZQuiz& game, const Evaluator& evaluator, int num_simulations, float epsilon, float alpha, Policy& policy) {
  // TODO: Implement MCTS, returning the generated `policy`.
  //
  // To run the neural network, use the given `evaluator`, which returns a policy and
  // a value function for the given game.
}
