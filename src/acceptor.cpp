#include "acceptor.hpp"
#include "metrics.hpp"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/event.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>

Acceptor::Acceptor(uint16_t port, MPMCQueue<Request, 4096>& queue, Metrics& metrics)
    : queue_(queue), metrics_(metrics), conn_bufs_(std::make_unique<ConnBuf[]>(kMaxFds)) {
    // Create listening socket
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        perror("socket");
        std::exit(1);
    }

    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        close(listen_fd_);
        std::exit(1);
    }

    if (listen(listen_fd_, 512) < 0) {
        perror("listen");
        close(listen_fd_);
        std::exit(1);
    }

    set_nonblocking(listen_fd_);

    // Create kqueue (macOS equivalent of epoll)
    kq_fd_ = kqueue();
    if (kq_fd_ < 0) {
        perror("kqueue");
        close(listen_fd_);
        std::exit(1);
    }

    // Register listen socket for read events
    struct kevent ev;
    EV_SET(&ev, listen_fd_, EVFILT_READ, EV_ADD | EV_CLEAR, 0, 0, nullptr);
    if (kevent(kq_fd_, &ev, 1, nullptr, 0, nullptr) < 0) {
        perror("kevent add listen");
        close(listen_fd_);
        close(kq_fd_);
        std::exit(1);
    }
}

Acceptor::~Acceptor() {
    if (kq_fd_ >= 0) close(kq_fd_);
    if (listen_fd_ >= 0) close(listen_fd_);
}

void Acceptor::set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) { perror("fcntl F_GETFL"); return; }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("fcntl F_SETFL");
    }
}

void Acceptor::run() {
    struct kevent events[kMaxEvents];

    while (running_.load(std::memory_order_relaxed)) {
        struct timespec timeout = {0, 100'000'000}; // 100ms
        int n = kevent(kq_fd_, nullptr, 0, events, kMaxEvents, &timeout);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("kevent wait");
            break;
        }

        for (int i = 0; i < n; ++i) {
            int fd = static_cast<int>(events[i].ident);
            if (events[i].flags & EV_ERROR) {
                std::cerr << "kevent error on fd " << fd << ": "
                          << strerror(static_cast<int>(events[i].data)) << "\n";
                if (fd != listen_fd_) close_fd(fd);
                continue;
            }
            if (fd == listen_fd_) {
                handle_new_connection();
            } else {
                handle_readable(fd);
            }
        }
    }
}

void Acceptor::stop() {
    running_.store(false, std::memory_order_relaxed);
}

void Acceptor::handle_new_connection() {
    for (;;) {
        struct sockaddr_in client_addr{};
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &addr_len);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }

        if (client_fd >= kMaxFds) {
            std::cerr << "fd " << client_fd << " exceeds kMaxFds\n";
            close(client_fd);
            continue;
        }

        set_nonblocking(client_fd);
        conn_bufs_[client_fd].len = 0;

        // Register with kqueue
        struct kevent ev;
        EV_SET(&ev, client_fd, EVFILT_READ, EV_ADD | EV_CLEAR, 0, 0, nullptr);
        if (kevent(kq_fd_, &ev, 1, nullptr, 0, nullptr) < 0) {
            perror("kevent add client");
            close(client_fd);
            continue;
        }
    }
}

void Acceptor::handle_readable(int client_fd) {
    ConnBuf& buf = conn_bufs_[client_fd];

    for (;;) {
        if (buf.len >= kRecvBufSize) {
            send_error(client_fd, 413, "Request too large");
            close_fd(client_fd);
            return;
        }

        ssize_t n = read(client_fd, buf.data + buf.len, kRecvBufSize - buf.len);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            if (errno == EINTR) continue;
            close_fd(client_fd);
            return;
        }
        if (n == 0) {
            // Client closed connection
            close_fd(client_fd);
            return;
        }
        buf.len += n;
    }

    // Try to parse the HTTP request
    Request req{};
    if (parse_http(buf.data, buf.len, req)) {
        req.fd = client_fd;
        req.arrival_ns = now_ns();

        // Check for /metrics endpoint
        if (buf.len >= 12 && strncmp(buf.data, "GET /metrics", 12) == 0) {
            std::string json = metrics_.metrics_json();
            char resp[1024];
            int resp_len = snprintf(resp, sizeof(resp),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %zu\r\n"
                "Connection: close\r\n"
                "\r\n%s",
                json.size(), json.c_str());
            send_response(client_fd, resp, resp_len);
            close_fd(client_fd);
            return;
        }

        if (!queue_.push(std::move(req))) {
            send_error(client_fd, 503, "Server overloaded");
            close_fd(client_fd);
        }
        // fd ownership transferred to queue consumer; don't close here
        buf.len = 0;
    }
    // else: incomplete request, wait for more data
}

bool Acceptor::parse_http(const char* buf, size_t len, Request& req) {
    // Find end of headers
    const char* header_end = nullptr;
    for (size_t i = 0; i + 3 < len; ++i) {
        if (buf[i] == '\r' && buf[i+1] == '\n' && buf[i+2] == '\r' && buf[i+3] == '\n') {
            header_end = buf + i + 4;
            break;
        }
    }
    if (!header_end) return false; // headers incomplete

    // Check for GET /metrics before POST check
    if (len >= 3 && strncmp(buf, "GET", 3) == 0) {
        return true; // let handle_readable deal with it
    }

    // Must be POST
    if (len < 4 || strncmp(buf, "POST", 4) != 0) {
        return false;
    }

    // Find Content-Length
    int content_length = 0;
    const char* cl = strcasestr(buf, "Content-Length:");
    if (cl && cl < header_end) {
        content_length = atoi(cl + 15);
    }

    if (content_length < 0 || static_cast<size_t>(content_length) > MAX_BODY_SIZE) {
        return false;
    }

    size_t header_size = header_end - buf;
    size_t total_needed = header_size + content_length;
    if (len < total_needed) return false; // body incomplete

    req.body_len = content_length;
    memcpy(req.body, header_end, content_length);
    return true;
}

void Acceptor::send_response(int fd, const char* data, size_t len) {
    size_t written = 0;
    while (written < len) {
        ssize_t n = write(fd, data + written, len - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            perror("write response");
            return;
        }
        written += n;
    }
}

void Acceptor::send_error(int fd, int status_code, const char* msg) {
    char body[256];
    int body_len = snprintf(body, sizeof(body), R"({"error":"%s"})", msg);

    const char* status_text = (status_code == 503) ? "Service Unavailable" : "Bad Request";
    char resp[512];
    int resp_len = snprintf(resp, sizeof(resp),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n%s",
        status_code, status_text, body_len, body);

    send_response(fd, resp, resp_len);
}

void Acceptor::close_fd(int fd) {
    if (fd >= 0 && fd < kMaxFds) {
        conn_bufs_[fd].len = 0;
    }
    close(fd);
}
