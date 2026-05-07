// This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <array>
#include <functional>

#include "board_game.h"

template<BoardGame G>
using Policy = std::array<float, G::ACTIONS>;

template<BoardGame G>
using Evaluator = std::function<void(const G&, Policy<G>&, float&)>;

template<BoardGame G>
void mcts(const G& game, const Evaluator<G>& evaluator, int num_simulations, float epsilon, float alpha, Policy<G>& policy) {
  // TODO: Implement MCTS, returning the generated `policy`.
  //
  // To run the neural network, use the given `evaluator`, which returns a policy and
  // a value function for the given game.
}
