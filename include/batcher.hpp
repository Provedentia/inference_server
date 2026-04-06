#pragma once

#include "acceptor.hpp"
#include "metrics.hpp"

#include <mutex>
#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include <cstddef>

struct BatchConfig {
    size_t   max_batch_size = 32;
    uint32_t timeout_us     = 500;
    uint32_t inference_us   = 1000;   // simulated inference time
};

class Batcher {
public:
    explicit Batcher(BatchConfig cfg, Metrics& metrics);
    ~Batcher();

    // Called from worker threads — must be thread-safe.
    void add(Request&& req);

private:
    bool should_flush() const;  // caller must hold mu_
    void flush();               // caller must hold mu_
    void timer_loop();          // background thread for timeout flushes

    BatchConfig              cfg_;
    Metrics&                 metrics_;
    std::mutex               mu_;
    std::vector<Request>     pending_;
    uint64_t                 batch_start_ns_{0};
    std::atomic<bool>        running_{true};
    std::thread              timer_thread_;
};
