#include "batcher.hpp"
#include "metrics.hpp"

#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cstring>

// Create a connected socket pair — write end acts as client fd
static std::pair<int, int> make_socket_pair() {
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) < 0) {
        perror("socketpair");
        std::exit(1);
    }
    return {fds[0], fds[1]};
}

// Read response from the read end of a socket pair
static std::string read_response(int fd, int timeout_ms = 500) {
    char buf[1024] = {};
    // Use poll-style wait
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    ssize_t total = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        ssize_t n = read(fd, buf + total, sizeof(buf) - total - 1);
        if (n > 0) {
            total += n;
            break; // got data
        }
        if (n == 0) break; // closed
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        break;
    }
    return std::string(buf, total);
}

static void test_timeout_flush() {
    std::cout << "Test 1: Timeout flush... ";

    Metrics metrics;
    BatchConfig cfg;
    cfg.max_batch_size = 100;  // large — won't trigger size flush
    cfg.timeout_us = 100;      // 100us timeout
    cfg.inference_us = 0;      // no simulated inference

    Batcher batcher(cfg, metrics);

    // Send 5 requests slowly, one per 50ms
    std::vector<std::pair<int, int>> socket_pairs;
    for (int i = 0; i < 5; ++i) {
        auto [read_end, write_end] = make_socket_pair();
        socket_pairs.push_back({read_end, write_end});

        Request req{};
        req.fd = write_end;
        req.arrival_ns = now_ns();

        // Small delay so timeout can fire between adds
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        batcher.add(std::move(req));
    }

    // Wait a bit for timeout flushes
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check that all requests got responses
    int responses = 0;
    for (auto& [read_end, write_end] : socket_pairs) {
        std::string resp = read_response(read_end, 200);
        if (!resp.empty() && resp.find("200 OK") != std::string::npos) {
            responses++;
        }
        close(read_end);
        // write_end closed by batcher
    }

    assert(responses == 5 && "Not all requests received responses via timeout flush");
    std::cout << "PASS (" << responses << "/5 responses)\n";
}

static void test_size_flush() {
    std::cout << "Test 2: Size flush... ";

    Metrics metrics;
    BatchConfig cfg;
    cfg.max_batch_size = 8;
    cfg.timeout_us = 10'000'000; // 10s — should never fire
    cfg.inference_us = 0;

    Batcher batcher(cfg, metrics);

    auto start = std::chrono::steady_clock::now();

    std::vector<std::pair<int, int>> socket_pairs;
    for (int i = 0; i < 8; ++i) {
        auto [read_end, write_end] = make_socket_pair();
        socket_pairs.push_back({read_end, write_end});

        Request req{};
        req.fd = write_end;
        req.arrival_ns = now_ns();
        batcher.add(std::move(req));
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // Check all got responses quickly (size flush should fire immediately on 8th add)
    int responses = 0;
    for (auto& [read_end, write_end] : socket_pairs) {
        std::string resp = read_response(read_end, 50);
        if (!resp.empty() && resp.find("200 OK") != std::string::npos) {
            responses++;
        }
        close(read_end);
    }

    assert(responses == 8 && "Not all requests received responses via size flush");
    assert(elapsed_ms < 50 && "Size flush took too long");
    std::cout << "PASS (" << responses << "/8 responses in " << elapsed_ms << "ms)\n";
}

int main() {
    test_timeout_flush();
    test_size_flush();
    std::cout << "All batcher tests passed.\n";
    return 0;
}
