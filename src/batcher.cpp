#include "batcher.hpp"

#include <unistd.h>
#include <sys/socket.h>
#include <cstdio>
#include <cstring>
#include <chrono>

Batcher::Batcher(BatchConfig cfg, Metrics& metrics)
    : cfg_(cfg), metrics_(metrics) {
    pending_.reserve(cfg_.max_batch_size);
    timer_thread_ = std::thread(&Batcher::timer_loop, this);
}

Batcher::~Batcher() {
    running_.store(false, std::memory_order_relaxed);
    if (timer_thread_.joinable()) timer_thread_.join();
    // Flush any remaining pending requests
    std::lock_guard<std::mutex> lock(mu_);
    if (!pending_.empty()) flush();
}

bool Batcher::should_flush() const {
    if (pending_.empty()) return false;
    if (pending_.size() >= cfg_.max_batch_size) return true;
    uint64_t elapsed_us = (now_ns() - batch_start_ns_) / 1000;
    return elapsed_us >= cfg_.timeout_us;
}

void Batcher::add(Request&& req) {
    std::lock_guard<std::mutex> lock(mu_);
    if (pending_.empty()) {
        batch_start_ns_ = now_ns();
    }
    pending_.push_back(std::move(req));

    if (should_flush()) {
        flush();
    }
}

void Batcher::timer_loop() {
    while (running_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::microseconds(cfg_.timeout_us / 2 + 1));
        std::lock_guard<std::mutex> lock(mu_);
        if (should_flush()) {
            flush();
        }
    }
}

void Batcher::flush() {
    if (pending_.empty()) return;

    uint64_t flush_start = now_ns();
    size_t batch_size = pending_.size();

    // Record batch_start_ns on each request
    for (auto& req : pending_) {
        req.batch_start_ns = flush_start;
    }

    // Simulate inference
    if (cfg_.inference_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(cfg_.inference_us));
    }

    uint64_t inference_done = now_ns();

    // Build response template
    char resp_body[128];
    int body_len = snprintf(resp_body, sizeof(resp_body),
        R"({"result":"ok","batch_size":%zu})", batch_size);

    char http_resp[256];
    int http_len = snprintf(http_resp, sizeof(http_resp),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n%s",
        body_len, resp_body);

    for (auto& req : pending_) {
        // Send response
        ssize_t written = 0;
        while (written < http_len) {
            ssize_t n = write(req.fd, http_resp + written, http_len - written);
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            written += n;
        }

        // Record metrics
        uint64_t end = now_ns();
        LatencySample sample{};
        sample.queue_wait_ns = req.queue_wait_ns;
        sample.batch_wait_ns = flush_start - req.arrival_ns - req.queue_wait_ns;
        sample.inference_ns  = inference_done - flush_start;
        sample.end_to_end_ns = end - req.arrival_ns;
        metrics_.record(sample);

        close(req.fd);
    }

    pending_.clear();
    batch_start_ns_ = 0;
}
