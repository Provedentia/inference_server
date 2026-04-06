#include "mpmc_queue.hpp"

#include <benchmark/benchmark.h>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>

// --- Lock-free MPMC: single producer, single consumer ---
static void BM_MPMCQueue_Throughput(benchmark::State& state) {
    for (auto _ : state) {
        MPMCQueue<uint64_t, 8192> queue;
        constexpr size_t kItems = 1'000'000;
        std::atomic<bool> done{false};

        std::thread consumer([&]() {
            uint64_t val;
            size_t count = 0;
            while (count < kItems) {
                if (queue.pop(val)) ++count;
            }
        });

        for (size_t i = 0; i < kItems; ++i) {
            uint64_t v = i;
            while (!queue.push(std::move(v))) {}
        }

        consumer.join();
        state.SetItemsProcessed(kItems);
    }
}
BENCHMARK(BM_MPMCQueue_Throughput);

// --- Lock-free MPMC: 4 producers, 4 consumers ---
static void BM_MPMCQueue_Contention(benchmark::State& state) {
    for (auto _ : state) {
        MPMCQueue<uint64_t, 8192> queue;
        constexpr size_t kProducers = 4;
        constexpr size_t kConsumers = 4;
        constexpr size_t kPerProd = 250'000;
        constexpr size_t kTotal = kProducers * kPerProd;
        std::atomic<uint64_t> consumed{0};

        std::vector<std::thread> consumers;
        for (size_t c = 0; c < kConsumers; ++c) {
            consumers.emplace_back([&]() {
                uint64_t val;
                while (consumed.load(std::memory_order_relaxed) < kTotal) {
                    if (queue.pop(val))
                        consumed.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }

        std::vector<std::thread> producers;
        for (size_t p = 0; p < kProducers; ++p) {
            producers.emplace_back([&, p]() {
                for (size_t i = 0; i < kPerProd; ++i) {
                    uint64_t v = p * kPerProd + i;
                    while (!queue.push(std::move(v))) {}
                }
            });
        }

        for (auto& t : producers) t.join();
        for (auto& t : consumers) t.join();
        state.SetItemsProcessed(kTotal);
    }
}
BENCHMARK(BM_MPMCQueue_Contention);

// --- Mutex queue: single producer, single consumer ---
static void BM_MutexQueue_Throughput(benchmark::State& state) {
    for (auto _ : state) {
        std::queue<uint64_t> q;
        std::mutex mu;
        constexpr size_t kItems = 1'000'000;

        std::thread consumer([&]() {
            size_t count = 0;
            while (count < kItems) {
                std::lock_guard<std::mutex> lock(mu);
                if (!q.empty()) {
                    q.pop();
                    ++count;
                }
            }
        });

        for (size_t i = 0; i < kItems; ++i) {
            std::lock_guard<std::mutex> lock(mu);
            q.push(i);
        }

        consumer.join();
        state.SetItemsProcessed(kItems);
    }
}
BENCHMARK(BM_MutexQueue_Throughput);

// --- Mutex queue: 4 producers, 4 consumers ---
static void BM_MutexQueue_Contention(benchmark::State& state) {
    for (auto _ : state) {
        std::queue<uint64_t> q;
        std::mutex mu;
        constexpr size_t kProducers = 4;
        constexpr size_t kConsumers = 4;
        constexpr size_t kPerProd = 250'000;
        constexpr size_t kTotal = kProducers * kPerProd;
        std::atomic<uint64_t> produced{0};
        std::atomic<uint64_t> consumed{0};

        std::vector<std::thread> consumers;
        for (size_t c = 0; c < kConsumers; ++c) {
            consumers.emplace_back([&]() {
                while (consumed.load(std::memory_order_relaxed) < kTotal) {
                    std::lock_guard<std::mutex> lock(mu);
                    if (!q.empty()) {
                        q.pop();
                        consumed.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }

        std::vector<std::thread> producers;
        for (size_t p = 0; p < kProducers; ++p) {
            producers.emplace_back([&, p]() {
                for (size_t i = 0; i < kPerProd; ++i) {
                    std::lock_guard<std::mutex> lock(mu);
                    q.push(p * kPerProd + i);
                }
            });
        }

        for (auto& t : producers) t.join();
        for (auto& t : consumers) t.join();
        state.SetItemsProcessed(kTotal);
    }
}
BENCHMARK(BM_MutexQueue_Contention);

BENCHMARK_MAIN();
