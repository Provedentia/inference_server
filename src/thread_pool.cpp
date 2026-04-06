#include "thread_pool.hpp"

#include <iostream>

ThreadPool::ThreadPool(size_t num_threads, MPMCQueue<Request, 4096>& queue, Batcher& batcher)
    : stats_(num_threads), queue_(queue), batcher_(batcher) {
    threads_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&ThreadPool::worker, this, i);
    }
}

ThreadPool::~ThreadPool() {
    stop();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::stop() {
    running_.store(false, std::memory_order_relaxed);
}

ThreadStats ThreadPool::aggregate_stats() const {
    ThreadStats agg{};
    for (auto& s : stats_) {
        agg.requests_processed += s.requests_processed;
        agg.total_queue_wait_ns += s.total_queue_wait_ns;
    }
    return agg;
}

void ThreadPool::worker(size_t thread_id) {
    // Note: pthread_setaffinity_np is Linux-only. On macOS, thread affinity
    // is managed by the kernel. We skip pinning on macOS.
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id % std::thread::hardware_concurrency(), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

    Request req{};
    while (running_.load(std::memory_order_relaxed)) {
        if (queue_.pop(req)) {
            uint64_t dequeue_time = now_ns();
            req.queue_wait_ns = dequeue_time - req.arrival_ns;

            stats_[thread_id].requests_processed++;
            stats_[thread_id].total_queue_wait_ns += req.queue_wait_ns;

            batcher_.add(std::move(req));
        } else {
            // No work — brief yield to avoid burning CPU
            std::this_thread::yield();
        }
    }

    // Drain remaining items on shutdown
    while (queue_.pop(req)) {
        uint64_t dequeue_time = now_ns();
        req.queue_wait_ns = dequeue_time - req.arrival_ns;
        stats_[thread_id].requests_processed++;
        stats_[thread_id].total_queue_wait_ns += req.queue_wait_ns;
        batcher_.add(std::move(req));
    }
}
