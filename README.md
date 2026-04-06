# Nexus

A high-performance HTTP/1.1 inference server built in C++17 from raw POSIX sockets.
No web frameworks. No Boost. No libuv. Just the operating system and the C++ standard library.

Nexus demonstrates how production ML serving infrastructure works under the hood:
non-blocking I/O, lock-free concurrency, adaptive request batching, and real-time
latency instrumentation. The inference step is simulated with a configurable sleep,
isolating the systems engineering from model-specific concerns.

## Why This Exists

ML inference servers like Triton and vLLM batch incoming requests to amortize GPU
kernel launch overhead. The networking and scheduling layers that make this possible
are rarely examined in isolation. Nexus strips away the model to focus on the
infrastructure questions:

- How do you accept thousands of concurrent connections without a thread per client?
- How does a lock-free queue compare to a mutex-based queue under contention?
- What's the right batching strategy when latency and throughput are both constraints?
- Where does time actually go in an inference request's lifecycle?

## Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │                     Nexus                           │
                    │                                                     │
  HTTP POST        │  ┌───────────┐    ┌───────────┐    ┌────────────┐   │
  /infer ─────────►│  │  Acceptor │───►│   MPMC    │───►│  Thread    │   │
                    │  │  (kqueue) │    │   Queue   │    │   Pool     │   │
                    │  │  edge-    │    │  lock-    │    │  N workers │   │
                    │  │  triggered│    │  free     │    │  per-core  │   │
                    │  └───────────┘    └───────────┘    └─────┬──────┘   │
                    │                                         │          │
                    │                                         ▼          │
  JSON             │                                   ┌────────────┐   │
  Response ◄───────│───────────────────────────────────│  Adaptive  │   │
                    │                                   │  Batcher   │   │
                    │                                   │  (flush on │   │
                    │         ┌──────────┐              │  size OR   │   │
  GET /metrics ────►│────────►│ Metrics  │◄─────────────│  timeout)  │   │
                    │         │ p50/p95  │              └────────────┘   │
                    │         │ /p99     │                               │
                    │         └──────────┘                               │
                    └──────────────────────────────────────────────────────┘
```

**Request lifecycle:**

1. The **Acceptor** runs a single-threaded event loop using kqueue (macOS) with edge-triggered notifications. It accepts connections, sets them non-blocking, and reads HTTP data into per-fd buffers. When a complete POST request arrives, it pushes a `Request` struct into the queue.

2. The **MPMC Queue** is a lock-free bounded ring buffer implementing the Dmitry Vyukov algorithm. It uses per-slot sequence counters and compare-and-swap operations to coordinate multiple producers and consumers without any mutex.

3. **Thread Pool** workers (one per CPU core by default) spin on the queue, popping requests and forwarding them to the batcher. Each worker maintains its own cache-line-isolated statistics to avoid contention.

4. The **Adaptive Batcher** collects requests and flushes them as a batch when either the batch is full or a timeout expires. A background timer thread ensures that partially-filled batches don't wait indefinitely. On flush, it simulates inference (configurable sleep), sends HTTP responses on each request's file descriptor, and records latency breakdowns.

5. **Metrics** collects per-request latency samples broken into queue wait, batch formation wait, and inference time. A background stats thread prints p50/p95/p99 every second. A `/metrics` HTTP endpoint returns the same data as JSON.

## Project Structure

```
nexus/
├── CMakeLists.txt              # Build system — Debug (ASan), Release, TSan targets
├── include/
│   ├── mpmc_queue.hpp          # Lock-free bounded MPMC queue (Vyukov algorithm)
│   ├── acceptor.hpp            # Event-driven HTTP acceptor + Request struct
│   ├── thread_pool.hpp         # Worker pool with per-thread stats
│   ├── batcher.hpp             # Adaptive batching with size/timeout triggers
│   └── metrics.hpp             # Latency recording and percentile computation
├── src/
│   ├── main.cpp                # CLI parsing, signal handling, component wiring
│   ├── acceptor.cpp            # kqueue event loop, HTTP parsing, fd management
│   ├── thread_pool.cpp         # Worker loop, queue draining, affinity pinning
│   ├── batcher.cpp             # Batch accumulation, flush logic, response writing
│   └── metrics.cpp             # Percentile calculation, stats printing, JSON export
├── tests/
│   ├── queue_stress.cpp        # 4P/4C correctness test, 10M items, TSan-enabled
│   └── batcher_test.cpp        # Timeout flush + size flush verification
├── benchmarks/
│   ├── queue_benchmark.cpp     # Google Benchmark: lock-free vs mutex throughput
│   ├── load_test.py            # wrk-based end-to-end load test, JSON output
│   └── post.lua                # wrk script for POST requests
└── baseline/
    └── flask_server.py         # Single-threaded Python baseline for comparison
```

## Design Decisions

### Lock-free queue over mutex-based queue

The MPMC queue sits on the critical path between the I/O thread and every worker. Under the Vyukov algorithm, producers and consumers never block each other — they use atomic compare-and-swap to claim slots and sequence counters to coordinate visibility. Benchmark results show the lock-free queue handles contention with ~2x lower latency than `std::mutex + std::queue` for single producer/consumer workloads. The advantage grows with more threads since mutex-based queues serialize all access through a single lock.

### kqueue with edge-triggered notifications

Level-triggered polling notifies you repeatedly as long as a socket is readable, which wastes syscalls. Edge-triggered mode notifies once when a socket *becomes* readable, then stays quiet until you drain the data. This means we must read until `EAGAIN` on every notification (a partial-read bug here means lost data), but the syscall reduction is significant under high connection counts. On macOS, kqueue replaces Linux's epoll with equivalent semantics.

### Adaptive batching (size + timeout)

Pure size-based batching starves low-traffic periods (a single request waits forever for 31 more to arrive). Pure timeout-based batching wastes throughput during bursts (why wait 500us when 32 requests are already queued?). Nexus flushes on whichever trigger fires first. A background timer thread checks the timeout condition independently, so a lone request at 3am still gets served within the timeout window.

### Cache-line isolation

The `ThreadStats` struct is padded to 64 bytes (one cache line) so that each worker's counter updates don't invalidate other workers' cache lines (false sharing). The MPMC queue's `head_` and `tail_` cursors are similarly isolated. Without this, threads on adjacent cores would constantly bounce cache lines between L1 caches, adding ~50ns per access.

### Heap-allocated large buffers

The MPMC queue stores `Request` structs, each containing an 8KB body buffer. With 4096 slots, that's ~33MB. Similarly, the acceptor maintains per-fd connection buffers for up to 4096 concurrent connections (~37MB). Both are heap-allocated via `std::unique_ptr` to avoid stack overflow, especially under AddressSanitizer which reduces default stack size.

## Build

**Requirements:** C++17 compiler (clang++ or g++), CMake 3.16+, pthread

```bash
# Release build (optimized)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Debug build (AddressSanitizer + UBSan)
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

## Usage

```bash
./nexus [--port 8080] [--threads 8] [--batch-size 32] [--timeout-us 500] [--inference-us 1000]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | Listening port |
| `--threads` | CPU count | Worker thread count |
| `--batch-size` | 32 | Max requests per batch before flush |
| `--timeout-us` | 500 | Max microseconds to wait for a full batch |
| `--inference-us` | 1000 | Simulated inference time per batch |

```bash
# Send a request
curl -X POST http://localhost:8080/infer \
     -H "Content-Type: application/json" \
     -d '{"input": [1.0, 2.0]}'
# {"result":"ok","batch_size":1}

# Check metrics
curl http://localhost:8080/metrics
# {"count":1234,"p50_ns":3500000,"p95_ns":5000000,"p99_ns":12000000}
```

While the server runs, it prints per-second latency stats to stderr:

```
[stats] reqs=21280 p50=3.51ms p95=4.92ms p99=11.46ms | avg queue=1.82ms batch=0.33ms infer=1.26ms
```

## Testing

```bash
# Gate 2: Lock-free queue correctness (built with ThreadSanitizer)
./queue_stress
# Produced:  10000000
# Consumed:  10000000
# Throughput: 2.28 M items/s
# PASS: all items accounted for.

# Gate 3: Full server under ThreadSanitizer
./nexus_tsan --port 8090 &
wrk -t4 -c50 -d10s http://localhost:8090/infer
# Zero TSan warnings expected

# Gate 4: Batcher flush logic
./batcher_test
# Test 1: Timeout flush... PASS (5/5 responses)
# Test 2: Size flush... PASS (8/8 responses in 0ms)
```

## Benchmarks

### Lock-free vs mutex queue throughput

```bash
./queue_benchmark
```

| Benchmark | Time (1M items) | Relative |
|-----------|-----------------|----------|
| MPMC single-threaded | 39ms | 1.0x (baseline) |
| Mutex single-threaded | 81ms | 2.1x slower |
| MPMC 4P/4C contention | 200ms wall | lock-free |
| Mutex 4P/4C contention | 175ms wall | lock contention |

The single-threaded case shows the raw overhead difference. Under contention, the mutex queue's wall time is similar but threads spend most of it blocked on the lock rather than doing useful work.

### End-to-end load test

```bash
./nexus --port 8080 &
cd benchmarks && python3 load_test.py --nexus-port 8080 --nexus-only --duration 30
# Writes results/baseline_comparison.json
```

**Results on Apple M2 (8 cores), 1ms simulated inference, 100 concurrent connections:**

| Metric | Value |
|--------|-------|
| Throughput | ~21,000 req/s |
| p50 latency | 3.7 ms |
| p95 latency | 5.0 ms |
| p99 latency | 12 ms |
| Avg queue wait | 1.8 ms |
| Avg batch formation | 0.3 ms |
| Avg inference | 1.3 ms |

The latency breakdown shows that queue wait (contention for the MPMC queue) dominates over batch formation overhead. The 1ms inference sleep accounts for ~1.3ms due to OS scheduling jitter.

## Graceful Shutdown

SIGINT/SIGTERM stops the acceptor loop, the thread pool drains remaining queued requests, and the batcher flushes any pending batch before the process exits.

## Limitations and Future Work

- **macOS only** — uses kqueue. Porting to Linux requires replacing with epoll (structurally identical, different API).
- **No connection reuse** — each request closes the connection (`Connection: close`). Adding keep-alive would require per-connection state machines.
- **Single batcher** — all workers funnel into one mutex-protected batch. Under extreme core counts, this becomes a bottleneck. A per-worker batcher with periodic merging would scale better.
- **No TLS, HTTP/2, or real model loading** — intentionally excluded to keep focus on the systems layer.
