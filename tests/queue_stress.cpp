#include "mpmc_queue.hpp"

#include <thread>
#include <vector>
#include <atomic>
#include <iostream>
#include <chrono>
#include <cassert>
#include <numeric>

static constexpr size_t kProducers    = 4;
static constexpr size_t kConsumers    = 4;
static constexpr size_t kItemsPerProd = 2'500'000;
static constexpr size_t kTotalItems   = kProducers * kItemsPerProd;

int main() {
    MPMCQueue<uint64_t, 8192> queue;

    std::atomic<uint64_t> produced{0};
    std::atomic<uint64_t> consumed{0};
    std::atomic<bool>     done{false};

    // Each consumer accumulates its own count to avoid contention on a shared counter
    // during the hot loop.
    std::vector<uint64_t> per_consumer(kConsumers, 0);

    auto t_start = std::chrono::steady_clock::now();

    // --- Producers ---
    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (size_t p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p]() {
            for (size_t i = 0; i < kItemsPerProd; ++i) {
                uint64_t val = p * kItemsPerProd + i;
                while (!queue.push(std::move(val))) {
                    // Queue full — spin. In a real system we'd back off.
                }
            }
            produced.fetch_add(kItemsPerProd, std::memory_order_relaxed);
        });
    }

    // --- Consumers ---
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumers);
    for (size_t c = 0; c < kConsumers; ++c) {
        consumers.emplace_back([&, c]() {
            uint64_t local_count = 0;
            uint64_t val;
            while (!done.load(std::memory_order_relaxed) || queue.size_approx() > 0) {
                if (queue.pop(val)) {
                    ++local_count;
                } else {
                    // Brief pause to reduce contention when queue is empty
                    std::this_thread::yield();
                }
            }
            // Drain any remaining items after done flag is set
            while (queue.pop(val)) {
                ++local_count;
            }
            per_consumer[c] = local_count;
        });
    }

    // Wait for all producers to finish
    for (auto& t : producers) t.join();

    // Signal consumers that production is complete
    done.store(true, std::memory_order_relaxed);

    for (auto& t : consumers) t.join();

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    uint64_t total_consumed = std::accumulate(per_consumer.begin(), per_consumer.end(), uint64_t{0});

    std::cout << "Produced:  " << kTotalItems << "\n";
    std::cout << "Consumed:  " << total_consumed << "\n";
    std::cout << "Elapsed:   " << elapsed_s << " s\n";
    std::cout << "Throughput: " << static_cast<double>(kTotalItems) / elapsed_s / 1e6
              << " M items/s\n";

    assert(total_consumed == kTotalItems && "MISMATCH: consumed != produced");
    std::cout << "PASS: all items accounted for.\n";
    return 0;
}
