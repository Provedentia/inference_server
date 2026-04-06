#include "acceptor.hpp"
#include "mpmc_queue.hpp"
#include "thread_pool.hpp"
#include "batcher.hpp"
#include "metrics.hpp"

#include <iostream>
#include <thread>
#include <csignal>
#include <cstdlib>
#include <cstring>

// Global flag + acceptor pointer for signal handler (only mutable globals allowed)
static std::atomic<bool> g_running{true};
static Acceptor* g_acceptor = nullptr;

static void signal_handler(int) {
    g_running.store(false, std::memory_order_relaxed);
    if (g_acceptor) g_acceptor->stop();
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " [--port N] [--threads N] [--batch-size N]"
              << " [--timeout-us N] [--inference-us N]\n";
}

int main(int argc, char* argv[]) {
    uint16_t port = 8080;
    size_t num_threads = std::thread::hardware_concurrency();
    BatchConfig batch_cfg;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = static_cast<uint16_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_cfg.max_batch_size = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--timeout-us") == 0 && i + 1 < argc) {
            batch_cfg.timeout_us = static_cast<uint32_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--inference-us") == 0 && i + 1 < argc) {
            batch_cfg.inference_us = static_cast<uint32_t>(atoi(argv[++i]));
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    MPMCQueue<Request, 4096> queue;
    Metrics metrics;
    Batcher batcher{batch_cfg, metrics};
    ThreadPool pool{num_threads, queue, batcher};
    Acceptor acceptor{port, queue, metrics};
    g_acceptor = &acceptor;

    // Install signal handler for graceful shutdown
    struct sigaction sa{};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // Stats thread — prints latency breakdown every second
    std::thread stats_thread([&metrics]() {
        while (g_running.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            metrics.print_stats();
        }
    });
    stats_thread.detach();

    std::cout << "Nexus running on port " << port
              << " with " << num_threads << " workers"
              << " (batch=" << batch_cfg.max_batch_size
              << " timeout=" << batch_cfg.timeout_us << "us"
              << " inference=" << batch_cfg.inference_us << "us)\n";

    acceptor.run();  // blocks until SIGINT calls acceptor.stop()

    // ThreadPool destructor joins workers and drains queue
    return 0;
}
