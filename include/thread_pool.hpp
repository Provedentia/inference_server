#pragma once

#include "mpmc_queue.hpp"
#include "acceptor.hpp"
#include "batcher.hpp"

#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cstddef>

struct alignas(64) ThreadStats {
    uint64_t requests_processed{0};
    uint64_t total_queue_wait_ns{0};
    char     _pad[64 - 2 * sizeof(uint64_t)];
};

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads, MPMCQueue<Request, 4096>& queue, Batcher& batcher);
    ~ThreadPool();   // joins all threads

    ThreadStats aggregate_stats() const;
    void stop();

private:
    void worker(size_t thread_id);

    std::vector<std::thread>        threads_;
    std::vector<ThreadStats>        stats_;    // one per thread, cache-line isolated
    MPMCQueue<Request, 4096>&       queue_;
    Batcher&                        batcher_;
    std::atomic<bool>               running_{true};
};
