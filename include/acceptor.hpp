#pragma once

#include "mpmc_queue.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>

static constexpr size_t MAX_BODY_SIZE = 8192;

struct alignas(64) Request {
    int      fd;                       // client socket fd — write response here
    uint64_t arrival_ns;               // clock_gettime timestamp on accept
    uint64_t queue_wait_ns  = 0;       // filled by worker on dequeue
    uint64_t batch_start_ns = 0;       // filled by batcher on flush
    uint32_t body_len       = 0;
    char     body[MAX_BODY_SIZE] = {};
};

// Monotonic nanosecond clock.
inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
}

class Metrics; // forward declaration

class Acceptor {
public:
    explicit Acceptor(uint16_t port, MPMCQueue<Request, 4096>& queue, Metrics& metrics);
    ~Acceptor();

    void run();   // blocking — call from a dedicated thread
    void stop();  // thread-safe

private:
    static constexpr size_t kMaxEvents   = 64;
    static constexpr size_t kRecvBufSize = MAX_BODY_SIZE + 1024; // headers + body

    void handle_new_connection();
    void handle_readable(int client_fd);
    bool parse_http(const char* buf, size_t len, Request& req);
    void send_response(int fd, const char* json, size_t json_len);
    void send_error(int fd, int status_code, const char* msg);
    void close_fd(int fd);
    void set_nonblocking(int fd);

    int                          listen_fd_ = -1;
    int                          kq_fd_     = -1;  // kqueue fd (macOS)
    std::atomic<bool>            running_{true};
    MPMCQueue<Request, 4096>&    queue_;
    Metrics&                     metrics_;

    // Per-fd read buffer for partial reads (simple approach: one buffer per fd)
    // In production you'd use a slab allocator; for this study a map suffices.
    struct ConnBuf {
        char   data[kRecvBufSize] = {};
        size_t len = 0;
    };
    static constexpr int kMaxFds = 4096;
    std::unique_ptr<ConnBuf[]> conn_bufs_; // heap-allocated, indexed by fd
};
