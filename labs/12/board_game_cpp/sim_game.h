// This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "board_game.h"
#include "mcts.h"

template<BoardGame G>
using History = std::vector<std::tuple<G, Policy<G>, float>>;

template<BoardGame G>
using Batch = std::vector<std::tuple<const G*, Policy<G>*, float*>>;

template<BoardGame G>
using BatchEvaluator = std::function<void(const Batch<G>&)>;

class worker_shutdown_exception : public std::exception {};

template<BoardGame G>
class SimGame {
  public:
    void worker_thread(int num_simulations, int sampling_moves, float epsilon, float alpha) try {
      while (true) {
        auto history = std::make_unique<History<G>>();
        // TODO: Simulate one game, collecting all (Game, Policy, float) triples to `history`, where
        // - the `Policy` is the policy computed by `mcts`;
        // - the float value is the outcome of the whole game.
        // When calling `mcts`, you would like to use `this->worker_evaluator` as the evaluator;
        // however, `mcts` wants a function pointer as the `Evaluator<G>`, so you need to use
        // `std::bind_front(&SimGame::worker_evaluator, this)` as the second argument of `mcts`.

        // Once the whole game is finished, we pass it to processor to return it.
        {
          std::unique_lock processor_lock{processor_mutex};
          processor_result.push_back(std::move(history));
        }
        processor_cv.notify_one();
      }
    } catch (worker_shutdown_exception&) {
      return;
    }

    void simulated_games_start(int threads, int num_simulations, int sampling_moves, float epsilon, float alpha) {
      if (worker_shutdown)
        throw std::runtime_error("Cannot call simulated_games_start after simulated_games_stop was called.");

      worker_queue_limit = threads;
      for (int thread = 0; thread < threads; thread++)
        std::thread(&SimGame<G>::worker_thread, this, num_simulations, sampling_moves, epsilon, alpha).detach();
    }

    std::unique_ptr<History<G>> simulated_game(BatchEvaluator<G>& evaluator) {
      while (true) {
        std::unique_ptr<Batch<G>> batch;
        {
          std::unique_lock processor_lock{processor_mutex};
          processor_cv.wait(processor_lock, [this]{return processor_result.size() || processor_queue.size();});
          if (processor_result.size()) {
            auto result = std::move(processor_result.back());
            processor_result.pop_back();
            return result;
          }

          batch = std::move(processor_queue.back());
          processor_queue.pop_back();
        }

        evaluator(*batch);

        {
          std::unique_lock worker_lock{worker_mutex};
        }
        worker_cv.notify_all();
      }
    }

    void simulated_games_stop() {
      std::unique_lock worker_lock{worker_mutex};
      worker_shutdown = true;
      worker_cv.notify_all();
    }

  private:
    std::mutex worker_mutex;
    std::condition_variable worker_cv;
    Batch<G> worker_queue;
    size_t worker_queue_limit;

    bool worker_shutdown = false;

    std::mutex processor_mutex;
    std::condition_variable processor_cv;
    std::vector<std::unique_ptr<Batch<G>>> processor_queue;
    std::vector<std::unique_ptr<History<G>>> processor_result;

    void worker_evaluator(const G& game, Policy<G>& policy, float& value) {
      std::unique_lock worker_lock{worker_mutex};

      value = INFINITY;
      worker_queue.emplace_back(&game, &policy, &value);
      if (worker_queue.size() == worker_queue_limit) {
        auto batch = std::make_unique<Batch<G>>(worker_queue);
        worker_queue.clear();
        {
          std::unique_lock processor_lock{processor_mutex};
          processor_queue.push_back(std::move(batch));
        }
        processor_cv.notify_one();
      }

      if (!worker_shutdown)
        worker_cv.wait(worker_lock, [&value, this]{return std::isfinite(value) || worker_shutdown;});
      if (worker_shutdown) throw worker_shutdown_exception();
    }
};
