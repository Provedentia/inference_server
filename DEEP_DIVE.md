# Nexus Deep Dive — Interview Reference

This document explains every component of Nexus in enough detail that you can
walk through the design, justify each decision, and answer follow-up questions
in a systems design or coding interview. It assumes strong software engineering
experience but limited C++ familiarity.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [How a Request Flows Through the System](#2-how-a-request-flows-through-the-system)
3. [Component 1: The Lock-Free Queue (mpmc_queue.hpp)](#3-component-1-the-lock-free-queue)
4. [Component 2: The Acceptor (acceptor.hpp / acceptor.cpp)](#4-component-2-the-acceptor)
5. [Component 3: The Thread Pool (thread_pool.hpp / thread_pool.cpp)](#5-component-3-the-thread-pool)
6. [Component 4: The Adaptive Batcher (batcher.hpp / batcher.cpp)](#6-component-4-the-adaptive-batcher)
7. [Component 5: Metrics (metrics.hpp / metrics.cpp)](#7-component-5-metrics)
8. [Component 6: Main and Signal Handling (main.cpp)](#8-component-6-main-and-signal-handling)
9. [Memory Layout and Performance Considerations](#9-memory-layout-and-performance-considerations)
10. [Concurrency Model and Thread Safety](#10-concurrency-model-and-thread-safety)
11. [Testing Strategy](#11-testing-strategy)
12. [Likely Interview Questions and Answers](#12-likely-interview-questions-and-answers)

---

## 1. The Big Picture

Nexus is a systems study of ML inference serving infrastructure. In production,
servers like Triton Inference Server or vLLM accept HTTP requests containing
model inputs, batch multiple requests together (because GPUs are more efficient
processing batches), run inference, and return results. The batching and
scheduling layers are complex concurrent systems that are normally hidden inside
large frameworks.

Nexus isolates and implements just those layers:

- **Accept connections** without creating a thread per client
- **Queue requests** using a lock-free data structure
- **Batch requests** adaptively based on count and time
- **Measure latency** broken down by phase

The inference itself is a `sleep()` — the interesting part is everything around it.

### What This Demonstrates to an Interviewer

- You understand non-blocking I/O and event-driven architecture
- You can implement and reason about lock-free data structures
- You understand CPU cache behavior and how it affects concurrent code
- You can design adaptive systems that balance latency vs throughput
- You test concurrent code rigorously (sanitizers, stress tests)
- You measure performance rather than guessing

---

## 2. How a Request Flows Through the System

Follow one HTTP request from arrival to response. This is the most important
thing to be able to explain clearly.

```
1. Client sends:  POST /infer HTTP/1.1\r\nContent-Length: 22\r\n\r\n{"input":[1.0,2.0]}

2. KERNEL:        The OS kernel's TCP stack receives the SYN, completes the
                  three-way handshake, and places the new connection in the
                  listen socket's accept queue.

3. ACCEPTOR:      The kqueue event loop wakes up (the listen fd became readable).
                  accept() returns a new file descriptor for this connection.
                  The acceptor sets it non-blocking and registers it with kqueue.

4. ACCEPTOR:      kqueue fires again — the client fd is readable. The acceptor
                  reads data into a per-fd buffer. It scans for \r\n\r\n to find
                  the end of HTTP headers, extracts Content-Length, and reads
                  that many body bytes.

5. ACCEPTOR:      Constructs a Request struct:
                  - fd = the client's socket file descriptor
                  - arrival_ns = current monotonic timestamp
                  - body = the POST body bytes
                  Pushes it into the MPMC queue. If the queue is full, sends
                  HTTP 503 and closes the connection.

6. MPMC QUEUE:    The Request sits in a slot in the ring buffer. The slot's
                  sequence counter has been advanced to signal "data available."

7. WORKER:        One of N worker threads was spinning on queue.pop(). It
                  atomically claims the slot, moves the Request out, and
                  advances the slot's sequence counter to signal "slot free."
                  It records queue_wait_ns = now - arrival_ns.

8. BATCHER:       The worker calls batcher.add(request). The batcher locks its
                  mutex, appends the request to a pending vector. It checks:
                  - Is the batch full (>= max_batch_size)? Flush now.
                  - Has the timeout expired? Flush now.
                  - Neither? Return, the timer thread will flush later.

9. FLUSH:         When flush triggers (from add() or the timer thread):
                  - Record batch_start_ns on each request
                  - Sleep for inference_us microseconds (simulated GPU work)
                  - Build HTTP response: {"result":"ok","batch_size":N}
                  - Write response bytes to each request's fd
                  - Record latency sample (queue_wait, batch_wait, inference, e2e)
                  - Close each fd

10. CLIENT:       Receives HTTP 200 with JSON body. Connection closed.
```

**Key insight for interviews:** The file descriptor (fd) is the request's
identity throughout the system. It's how we know where to send the response.
Ownership of the fd transfers: acceptor creates it, pushes it into the queue,
worker pops it, batcher writes to it and closes it. At no point do two
components hold the fd simultaneously.

---

## 3. Component 1: The Lock-Free Queue

**File:** `include/mpmc_queue.hpp` (~107 lines)

This is the most technically interesting component and the most likely to get
deep interview questions.

### What "Lock-Free" Means

A lock-free data structure guarantees that *at least one thread* makes progress
in any given time window, even if other threads are paused or delayed. Compare:

- **Mutex-based:** If the thread holding the lock gets paused by the OS
  scheduler, every other thread blocks. One slow thread stalls everyone.
- **Lock-free:** Threads use atomic compare-and-swap (CAS) operations. If a CAS
  fails (another thread beat you), you retry. No thread can block another.

### The Vyukov Algorithm — How It Works

The queue is a fixed-size ring buffer where each slot has a **sequence counter**.
The sequence counter is the coordination mechanism that replaces locks.

```
Slots:     [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]
Sequences:  0    1    2    3    4    5    6    7     ← initial state
            ^                                        ← head (consumers read here)
            ^                                        ← tail (producers write here)
```

**Push (producer) algorithm:**

```
1. Read tail_ (the index where we want to write)
2. Look at slots[tail % capacity].sequence
3. Compare sequence with tail:
   - sequence == tail → Slot is free. Try to claim it:
     a. CAS tail_ from tail to tail+1
     b. If CAS succeeds: write data, set sequence = tail+1 (publish)
     c. If CAS fails: another producer beat us, retry from step 1
   - sequence < tail  → Slot still has unconsumed data. Queue is FULL.
   - sequence > tail  → Another producer claimed this slot. Reload tail, retry.
```

**Pop (consumer) algorithm:**

```
1. Read head_ (the index where we want to read)
2. Look at slots[head % capacity].sequence
3. Compare sequence with head+1:
   - sequence == head+1 → Data is ready. Try to claim it:
     a. CAS head_ from head to head+1
     b. If CAS succeeds: read data, set sequence = head+capacity (recycle)
     c. If CAS fails: another consumer beat us, retry from step 1
   - sequence < head+1 → No data yet. Queue is EMPTY.
   - sequence > head+1 → Another consumer took this slot. Reload head, retry.
```

### Why Sequence Counters Work

The sequence counter on each slot serves two purposes:

1. **Coordination:** It tells you whether a slot is ready for writing (free) or
   reading (contains data). No separate "full/empty" flag needed.
2. **ABA protection:** Because the sequence always increases (it's never reset),
   you can't confuse "slot was emptied and refilled" with "slot hasn't changed."
   In a naive CAS-on-data approach, a slot could be popped and pushed with the
   same value, and a stale CAS would incorrectly succeed. The monotonically
   increasing sequence prevents this.

### Memory Ordering — What It Means and Why It Matters

C++ has a concept called "memory ordering" that controls how atomic operations
are seen by other threads. On ARM chips (like Apple Silicon), the CPU can
reorder reads and writes for performance. Memory orderings prevent dangerous
reorderings.

Think of it like this: just because your code says "write A then write B"
doesn't mean another CPU core sees them in that order. Memory orderings add
fences that enforce visibility.

The orderings used in Nexus, from weakest to strongest:

| Ordering | What It Means | Used For |
|----------|---------------|----------|
| `relaxed` | No ordering guarantees. Other threads might see this write before earlier writes. | Reading cursors (head_, tail_) — we'll validate via the slot's sequence anyway |
| `acquire` | All reads/writes *after* this load are guaranteed to see everything *before* the corresponding release store. | Loading a slot's sequence — ensures we see the data that was written before the sequence was updated |
| `release` | All reads/writes *before* this store are guaranteed to be visible to any thread that does an acquire load of this value. | Storing a slot's sequence after writing data — publishes the data to consumers |

**The key pairing:** A producer does `release` store on the sequence after writing
data. A consumer does `acquire` load on the sequence before reading data. This
guarantees the consumer sees the data the producer wrote. Without this, on ARM,
the consumer could read stale/garbage data even though the sequence says "ready."

**Why not just use `sequential_consistency` (the strongest ordering) everywhere?**
Because it adds memory fence instructions on every atomic operation. On ARM, a
seq_cst store requires a `DMB ISH` instruction that stalls the pipeline. Relaxed
operations compile to plain loads/stores. The Vyukov algorithm is carefully
designed so that relaxed is sufficient for the cursors and acquire/release is
sufficient for the data publication. Using stronger orderings would add ~10-20ns
per operation — significant at millions of ops/second.

### The `static_assert` and Power-of-2 Requirement

```cpp
static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
static constexpr size_t kMask = Capacity - 1;
```

When capacity is a power of 2, `index % capacity` can be computed as
`index & (capacity - 1)`, which is a single AND instruction instead of an
expensive integer division. At millions of operations per second, this matters.

### Heap Allocation

The queue slots are heap-allocated via `std::unique_ptr<Slot[]>`:

```cpp
std::unique_ptr<Slot[]> slots_;
```

Each `Request` struct is ~8KB (due to the body buffer). With 4096 slots, that's
~33MB. Putting this on the stack would overflow it, especially under
AddressSanitizer which reduces stack size. `unique_ptr` ensures automatic cleanup
when the queue is destroyed (RAII).

---

## 4. Component 2: The Acceptor

**Files:** `include/acceptor.hpp`, `src/acceptor.cpp`

### Why Not "Thread Per Connection"

The naive approach to handling multiple clients is to spawn a thread for each
connection. Problems:

- Each thread costs ~8MB of stack space. 1000 connections = 8GB just for stacks.
- Context switching between thousands of threads is expensive.
- Thread creation/destruction has overhead.

The alternative: **event-driven I/O**. One thread monitors thousands of sockets
using an OS mechanism (kqueue on macOS, epoll on Linux) and reacts to events.

### How kqueue Works

kqueue is macOS's event notification mechanism (analogous to Linux's epoll):

```cpp
// Create a kqueue instance
int kq = kqueue();

// Register interest in a file descriptor
struct kevent ev;
EV_SET(&ev, fd, EVFILT_READ, EV_ADD | EV_CLEAR, 0, 0, nullptr);
kevent(kq, &ev, 1, nullptr, 0, nullptr);

// Wait for events (blocks until something happens or timeout)
struct kevent events[64];
int n = kevent(kq, nullptr, 0, events, 64, &timeout);
// n events are ready — process them
```

**EV_CLEAR** (edge-triggered mode): The kernel only notifies you when a socket
*transitions* from "no data" to "has data." After notification, you must drain
all available data (read until `EAGAIN`), or you won't be notified again until
*new* data arrives. This is more efficient than level-triggered mode (which
notifies repeatedly as long as data exists) but requires careful coding.

### Non-Blocking Sockets

```cpp
fcntl(fd, F_SETFL, flags | O_NONBLOCK);
```

A blocking `read()` on a socket with no data will suspend the thread. In an event
loop, that would freeze the entire server. Non-blocking mode makes `read()` return
immediately with `EAGAIN` if no data is available, so we can move on to the next
event.

### The HTTP Parser

This is intentionally minimal — we only need to handle `POST /infer` and
`GET /metrics`. The parser:

1. Scans for `\r\n\r\n` (the boundary between HTTP headers and body)
2. If not found, the request is incomplete — store what we have, wait for more data
3. Extracts `Content-Length` from headers
4. If we have `headers + Content-Length` bytes total, the request is complete
5. Copies the body into the `Request` struct

### Per-fd Buffers and Partial Reads

HTTP requests can arrive in fragments (TCP doesn't guarantee message boundaries).
A 1000-byte request might arrive as two 500-byte reads. The acceptor maintains
a buffer per file descriptor (`conn_bufs_[fd]`) to accumulate partial reads.

The buffer array is indexed directly by fd number (which the kernel assigns as
small integers). This is O(1) lookup — faster than a hash map.

### File Descriptor Ownership

This is a subtle but important design point. When a request is pushed to the queue,
the *acceptor stops managing that fd*. It doesn't close it, doesn't monitor it
with kqueue. The fd is now "owned" by whoever pops the request — ultimately the
batcher, which writes the response and closes it.

If the queue is full, the acceptor retains ownership: it sends a 503 error and
closes the fd itself.

---

## 5. Component 3: The Thread Pool

**Files:** `include/thread_pool.hpp`, `src/thread_pool.cpp`

### Worker Loop

Each worker thread runs this loop:

```
while (running) {
    if (queue.pop(request)) {
        request.queue_wait_ns = now - request.arrival_ns;
        batcher.add(request);
    } else {
        yield();  // no work, give CPU to other threads
    }
}
```

The `yield()` is important — without it, idle workers would spin at 100% CPU.
`std::this_thread::yield()` tells the OS scheduler "I have nothing to do, let
someone else run." This is a trade-off: yield adds latency on the next pop
(the thread must be rescheduled) but saves CPU when the queue is empty.

### Per-Thread Statistics

```cpp
struct alignas(64) ThreadStats {
    uint64_t requests_processed{0};
    uint64_t total_queue_wait_ns{0};
    char     _pad[64 - 2 * sizeof(uint64_t)];  // pad to exactly 64 bytes
};

std::vector<ThreadStats> stats_;  // one per thread
```

Each worker updates only `stats_[its_own_id]`. No locks needed because no two
threads touch the same element. The `alignas(64)` and padding ensure each
`ThreadStats` occupies exactly one cache line (see section 9).

### Thread Affinity (Linux Only)

```cpp
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id % hardware_concurrency(), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
```

This pins each worker to a specific CPU core. Benefits:
- The thread's data stays in that core's L1/L2 cache
- Avoids cache thrashing from the OS migrating threads between cores

macOS doesn't support `pthread_setaffinity_np` — the kernel manages thread
placement. This is a platform-specific optimization, not a correctness requirement.

### Graceful Drain on Shutdown

When `running_` becomes false, the worker doesn't just stop — it drains remaining
items from the queue. This prevents requests that were accepted but not yet
processed from being silently dropped.

---

## 6. Component 4: The Adaptive Batcher

**Files:** `include/batcher.hpp`, `src/batcher.cpp`

### Why Batching Matters

In real ML inference, running a batch of 32 inputs through a GPU takes almost
the same time as running 1 input. The GPU has thousands of cores that sit idle
with small batches. Batching amortizes:
- GPU kernel launch overhead
- Memory transfer latency
- Model weight loading

Even though Nexus simulates inference with sleep, the batching logic is real.

### The Two Flush Triggers

```cpp
bool should_flush() const {
    if (pending_.empty()) return false;
    if (pending_.size() >= cfg_.max_batch_size) return true;      // SIZE trigger
    uint64_t elapsed_us = (now_ns() - batch_start_ns_) / 1000;
    return elapsed_us >= cfg_.timeout_us;                          // TIMEOUT trigger
}
```

**Size trigger:** "We have enough requests, process them now."
- Maximizes throughput by using full batches.
- Example: max_batch_size=32, request rate=10K/s → batches fill in ~3ms.

**Timeout trigger:** "We've waited long enough, process what we have."
- Bounds worst-case latency for requests that arrive during quiet periods.
- Example: timeout_us=500 → no request waits more than 0.5ms for a batch.

### The Timer Thread Problem

The `add()` method checks `should_flush()` on every call. But what if 3 requests
arrive at once and then nothing for 10 seconds? The flush would never trigger
because no one is calling `add()`.

Solution: a background thread that periodically checks and flushes:

```cpp
void Batcher::timer_loop() {
    while (running_) {
        sleep_for(timeout_us / 2);  // check at 2x the timeout frequency
        lock(mu_);
        if (should_flush()) flush();
    }
}
```

The timer sleeps for half the timeout period, so the worst-case additional
latency is `timeout_us / 2`. This is the Nyquist-like principle: sample at
2x the frequency of the signal you're trying to catch.

### Mutex Usage

Unlike the MPMC queue, the batcher uses a plain `std::mutex`. Why?

- The batcher's critical section includes I/O (writing HTTP responses) and a
  sleep (simulated inference). These are millisecond-scale operations.
- Lock-free structures shine for short critical sections (nanoseconds).
  The overhead of CAS retry loops is wasted when the operation under the lock
  takes 1ms anyway.
- A mutex is simpler, easier to reason about, and correct.

**Key principle:** Use lock-free where contention is high and critical sections
are short (the queue). Use mutexes where operations are long and simplicity
matters (the batcher).

### The Flush Sequence

```
1. Record batch_start_ns on each request
2. Sleep for inference_us (simulated GPU work)
3. Build one HTTP response template (same for all requests in the batch)
4. For each request:
   a. Write the HTTP response to request.fd
   b. Record latency metrics
   c. Close request.fd
5. Clear the pending vector
```

Building the response template once and reusing it for all requests in the batch
is a small optimization — in a real system, each response would have different
model outputs.

---

## 7. Component 5: Metrics

**Files:** `include/metrics.hpp`, `src/metrics.cpp`

### Latency Breakdown

Each request produces a `LatencySample` with four timestamps:

```
|←—— queue_wait ——→|←—— batch_wait ——→|←—— inference ——→|
|                   |                   |                  |
arrival_ns     dequeue_time      batch_start_ns    inference_done
|←———————————————— end_to_end_ns ————————————————————————→|
```

- **queue_wait_ns:** Time spent in the MPMC queue waiting for a worker to pop it.
  High values = queue contention or not enough workers.
- **batch_wait_ns:** Time spent waiting in the batcher for the batch to fill or
  timeout to expire. High values = low request rate or batch size too large.
- **inference_ns:** The simulated inference time. Should be close to `inference_us`.
  Deviation = OS scheduling jitter.
- **end_to_end_ns:** Total time from accept to response sent.

### Percentile Computation

```cpp
void Metrics::print_stats() {
    vector<LatencySample> snap;
    {
        lock_guard lock(mu_);
        snap.swap(samples_);  // atomic swap, O(1)
    }
    // Now we own snap, no lock needed
    sort(snap);
    p50 = snap[0.50 * (n-1)];
    p95 = snap[0.95 * (n-1)];
    p99 = snap[0.99 * (n-1)];
}
```

The `swap` trick is important: we hold the lock only long enough to swap the
vector with an empty one (O(1) pointer swap). The actual sorting and computation
happens outside the lock, so we don't block `record()` calls from other threads.

In production systems, you'd use a streaming percentile estimator (t-digest,
HDR histogram) instead of sorting. Sorting is O(n log n) per stats window,
which is fine for our throughput (~20K samples/sec) but would break at millions.

---

## 8. Component 6: Main and Signal Handling

**File:** `src/main.cpp`

### Object Lifetime and Construction Order

```cpp
MPMCQueue<Request, 4096> queue;       // 1. queue created first
Metrics metrics;                       // 2. metrics (no dependencies)
Batcher batcher{batch_cfg, metrics};   // 3. batcher needs metrics
ThreadPool pool{num_threads, queue, batcher};  // 4. pool needs queue + batcher
Acceptor acceptor{port, queue, metrics};       // 5. acceptor needs queue + metrics
```

This order matters. C++ destroys objects in reverse order of construction:
acceptor first (stops accepting new connections), then pool (drains queue and
joins workers), then batcher (flushes remaining batch), then the rest. This
ensures graceful shutdown without use-after-free bugs.

### Signal Handling

```cpp
static std::atomic<bool> g_running{true};
static Acceptor* g_acceptor = nullptr;

static void signal_handler(int) {
    g_running.store(false, std::memory_order_relaxed);
    if (g_acceptor) g_acceptor->stop();
}
```

Signal handlers in C/C++ are extremely restricted — you can only safely call
"async-signal-safe" functions. You cannot allocate memory, lock mutexes, or call
most library functions. Writing to an atomic variable and calling `stop()` (which
just writes to another atomic) is safe.

The global `g_acceptor` pointer is the only global mutable state in the system
(besides `g_running`). It exists because signal handlers can't capture closures
or access local variables.

---

## 9. Memory Layout and Performance Considerations

### Cache Lines and False Sharing

A CPU cache line is 64 bytes on all modern x86/ARM chips. When one core writes
to any byte in a cache line, the *entire line* is invalidated in all other cores'
caches. This is called **false sharing** — two logically independent variables
on the same cache line cause cache invalidation ping-pong.

Example without padding:

```
Thread 0 writes stats[0].requests_processed  → invalidates line for thread 1
Thread 1 writes stats[1].requests_processed  → invalidates line for thread 0
Repeat millions of times → ~50ns penalty per access
```

With `alignas(64)`, each stat struct starts at a 64-byte boundary:

```
Cache line 0: [stats[0].requests_processed | stats[0].total_queue_wait_ns | padding]
Cache line 1: [stats[1].requests_processed | stats[1].total_queue_wait_ns | padding]
```

Now thread 0 and thread 1 never touch each other's cache lines.

The MPMC queue applies the same principle to `head_` and `tail_`:

```cpp
alignas(64) std::atomic<size_t> head_{0};  // consumers modify this
alignas(64) std::atomic<size_t> tail_{0};  // producers modify this
```

Producers never read `head_` and consumers never read `tail_`, so they operate
on independent cache lines.

### The Request Struct

```cpp
struct alignas(64) Request {
    int      fd;                       // 4 bytes
    uint64_t arrival_ns;               // 8 bytes
    uint64_t queue_wait_ns  = 0;       // 8 bytes
    uint64_t batch_start_ns = 0;       // 8 bytes
    uint32_t body_len       = 0;       // 4 bytes
    char     body[8192]     = {};      // 8192 bytes
};
// Total: ~8224 bytes per request
```

The `alignas(64)` ensures each Request starts at a cache line boundary, which
matters when Request structs are packed in the queue slots.

### Why `std::move` Everywhere

In C++, `std::move` doesn't actually move anything — it's a cast that says
"I'm done with this object, you can steal its guts." For a Request with an
8KB body buffer, move vs copy means:

- **Copy:** memcpy 8KB of body data → ~1us
- **Move:** For our struct (all primitive types and fixed-size array), move is
  the same as copy. But for types with heap allocations (like `std::string`,
  `std::vector`), move just transfers the pointer — O(1).

We use `std::move` consistently as a habit and because the queue's `push(T&&)`
signature requires it (rvalue reference).

---

## 10. Concurrency Model and Thread Safety

### Thread Map

| Thread | What It Does | Shared State |
|--------|-------------|--------------|
| Main thread | Runs acceptor event loop | queue_ (push), metrics_ (metrics endpoint) |
| Worker 0..N | Pop from queue, add to batcher | queue_ (pop), batcher_ (add), stats_[id] (write) |
| Batcher timer | Periodically check/flush batcher | batcher.mu_ (lock), batcher.pending_ |
| Stats thread | Print metrics every second | metrics.mu_ (lock), metrics.samples_ |

### Synchronization Mechanisms Used

| Shared Resource | Mechanism | Why This Choice |
|-----------------|-----------|----------------|
| MPMC queue slots | Lock-free (atomic CAS + sequence counters) | Hot path, nanosecond critical section, many contenders |
| Batcher pending vector | std::mutex | Millisecond critical section (includes I/O), few contenders |
| Metrics samples | std::mutex | Low frequency (per-request record, per-second read) |
| ThreadStats | No synchronization | Each thread only writes to its own slot |
| Acceptor running_ | std::atomic\<bool\> | Single writer (signal handler), single reader (event loop) |

### Potential Race Conditions and How They're Prevented

**Race 1: Two workers pop the same request**
Prevention: The CAS on `head_` in `pop()` ensures only one thread wins. Losers
see their CAS fail and retry with the updated head.

**Race 2: Batcher timer and worker both call flush()**
Prevention: Both acquire `mu_` before touching `pending_`. Only one can hold
the lock. The other blocks.

**Race 3: Close fd twice**
Prevention: fd ownership is linear — only the batcher calls `close(fd)`, and only
once per request (in `flush()`). The acceptor only closes fds it hasn't pushed
to the queue (503 errors, /metrics responses).

**Race 4: Signal handler and main thread**
Prevention: The signal handler only writes to `std::atomic<bool>` variables, which
is async-signal-safe. No other shared state is touched.

---

## 11. Testing Strategy

### Queue Stress Test (queue_stress.cpp)

**What it tests:** Correctness of the lock-free queue under heavy contention.

- 4 producer threads each push 2.5M items (10M total)
- 4 consumer threads pop until done
- Assertion: consumed count == 10M (no lost or duplicated items)
- Built with ThreadSanitizer (`-fsanitize=thread`)

**Why ThreadSanitizer matters:** TSan instruments every memory access and
detects data races at runtime. It catches races that would only manifest as
bugs under specific timing conditions (heisenbugs). The queue stress test
passes with zero TSan warnings, which is strong evidence of correctness.

**What TSan would catch:** If we had a memory ordering bug (e.g., used `relaxed`
where `acquire` was needed), TSan would report a data race between the producer's
write to `slot.data` and the consumer's read of `slot.data`.

### Batcher Test (batcher_test.cpp)

**Test 1 — Timeout flush:**
- Batch size = 100 (won't trigger), timeout = 100us
- Send 5 requests with 50ms gaps
- Assert: all 5 get responses (the timer thread flushed them)
- Validates that the background timer thread works correctly

**Test 2 — Size flush:**
- Batch size = 8, timeout = 10 seconds (won't trigger)
- Send 8 requests rapidly
- Assert: all 8 get responses within 50ms
- Validates that the size trigger works in the `add()` path

**Socket pairs:** The tests use `socketpair()` to create connected sockets in the
same process. The batcher writes to one end, the test reads from the other. This
avoids needing a running TCP server for unit tests.

### Full-Server TSan Test (Gate 3)

Run the server compiled with TSan (`nexus_tsan`) under wrk load:

```bash
./nexus_tsan &
wrk -t4 -c50 -d10s http://localhost:8090/infer
```

This exercises the entire pipeline — acceptor, queue, thread pool, batcher,
metrics — under realistic concurrent load. Zero TSan warnings means no data
races in the integrated system, not just in isolated components.

---

## 12. Likely Interview Questions and Answers

### "Why not just use a mutex-based queue?"

The queue is the central coordination point — every request passes through it,
and it's accessed by N+1 threads (N workers + 1 acceptor). With a mutex, only
one thread can access the queue at a time. If a worker is inside `pop()`, the
acceptor can't `push()`. Under high load, threads spend more time waiting for
the lock than doing work.

The lock-free queue allows one push and one pop to happen simultaneously with
zero blocking. Under the benchmark, the lock-free queue achieves ~2x throughput
for single producer/consumer.

That said, the batcher uses a mutex, and that's fine — its critical section
includes millisecond-scale I/O, so the lock overhead is negligible by comparison.
The right tool depends on the workload.

### "Walk me through what happens with memory ordering if you used relaxed everywhere"

If the producer's sequence store after writing data was `relaxed` instead of
`release`, the consumer could see the updated sequence ("data is ready") before
seeing the actual data. On x86 this accidentally works (x86 has a strong memory
model where stores are naturally ordered). On ARM (Apple Silicon, AWS Graviton),
the CPU can reorder the data write after the sequence write, causing the
consumer to read stale or zeroed-out data.

The `release` on the sequence store ensures: "everything I wrote before this
(the data) is visible to anyone who does an `acquire` load on this sequence."
The `acquire` on the consumer's sequence load ensures: "everything written
before the release store I just observed is visible to me."

### "What would you change to handle 100K concurrent connections?"

Three things:

1. **Replace per-fd buffer array with a hash map or slab allocator.** The current
   `ConnBuf[4096]` array limits us to fd numbers < 4096. With 100K connections,
   fd numbers would exceed that. A hash map scales but adds allocation overhead;
   a slab allocator gives O(1) allocation with better cache behavior.

2. **Multiple acceptor threads** or `SO_REUSEPORT` with multiple listening sockets.
   One event loop becomes a bottleneck beyond ~50K connections because `kevent()`
   processing is single-threaded.

3. **Increase queue capacity** and potentially use multiple queues (one per worker)
   to reduce contention on `head_` and `tail_`.

### "Why is the batcher a bottleneck and how would you fix it?"

All worker threads funnel into one `mutex`-protected batcher. Under high core
counts, threads serialize on the mutex. The fix is per-worker batchers: each
worker accumulates its own batch and flushes independently. This eliminates
mutex contention entirely at the cost of slightly smaller batches (each worker
sees 1/Nth of the traffic).

The tradeoff: per-worker batchers have higher throughput but lower batch
utilization (fewer requests per batch = less GPU efficiency). You'd tune
worker count and batch parameters together.

### "How do you know there are no memory leaks?"

Three mechanisms:

1. **RAII everywhere.** `unique_ptr` owns the queue slots and connection buffers.
   Vectors manage their own memory. When objects go out of scope, destructors
   free resources automatically.

2. **AddressSanitizer (ASan)** in the Debug build detects leaks, buffer overflows,
   use-after-free, and stack overflow at runtime with ~2x slowdown.

3. **File descriptor lifecycle.** Every fd is either closed by the acceptor (for
   errors and /metrics) or by the batcher (after sending the response). The
   batcher's `flush()` always closes every fd in the batch, and `pending_.clear()`
   ensures no dangling references.

### "Why build this from scratch instead of using a framework?"

The purpose is demonstrating understanding of the systems underneath frameworks.
Anyone can call `app.post('/infer', handler)` in Express or Flask. This project
shows that I understand:

- How `accept()`, `read()`, `write()`, and `close()` work at the syscall level
- How event loops multiplex connections without threads
- How lock-free data structures coordinate concurrent access
- How CPU caches affect multi-threaded performance
- How to test concurrent code with sanitizers

In production, I'd use a framework. But knowing what the framework does lets me
debug it, configure it correctly, and choose the right one.

### "What's the throughput bottleneck in this system?"

With 1ms simulated inference and batch size 32:

- **Theoretical max:** 32 requests / 1ms = 32,000 rps (batch fully utilized)
- **Measured:** ~21,000 rps

The gap is from:
1. **Batch underutilization:** Not every flush has 32 requests. During load ramp-up
   and timeout flushes, batches are smaller.
2. **Batcher mutex contention:** Workers block on the mutex while flush() does I/O.
3. **Queue wait:** Workers contend for the MPMC queue head, adding ~1.8ms average.

The latency breakdown from metrics confirms this:
```
avg queue=1.82ms  batch=0.33ms  infer=1.26ms
```

Queue wait is the largest non-inference component. Reducing worker count or
using per-worker batchers would help.

### "How would you add real GPU inference?"

Replace the `sleep_for(inference_us)` in `flush()` with:

1. Copy input data from the batch's Request bodies into a contiguous GPU input
   buffer (cudaMemcpy or mapped memory).
2. Launch the model forward pass (e.g., CUDA kernel, TensorRT inference).
3. Copy output data back.
4. Serialize per-request outputs into JSON responses.

The batching logic stays the same — the batcher already collects N requests
and processes them together. The key addition is managing GPU memory buffers
sized for `max_batch_size` inputs.
