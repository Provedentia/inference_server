#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

// Lock-free bounded MPMC queue using the Dmitry Vyukov algorithm.
// Each slot carries a sequence counter that coordinates producers and consumers
// without locks. The capacity must be a power of 2 for fast modulo via bitmask.

template<typename T, size_t Capacity>
class MPMCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static constexpr size_t kMask = Capacity - 1;

    struct alignas(64) Slot {
        std::atomic<size_t> sequence;
        T data;
    };

    // head_ and tail_ are on separate cache lines to avoid false sharing
    // between producers (tail_) and consumers (head_).
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    // Heap-allocated to avoid stack overflow with large T (e.g., Request with 8KB body)
    std::unique_ptr<Slot[]> slots_;

public:
    MPMCQueue() : slots_(std::make_unique<Slot[]>(Capacity)) {
        for (size_t i = 0; i < Capacity; ++i)
            // relaxed: no other thread can observe slots_ during construction
            slots_[i].sequence.store(i, std::memory_order_relaxed);
    }

    // Returns false if queue is full.
    bool push(T&& val) {
        // relaxed: we will validate via the slot's sequence; no ordering needed yet
        size_t tail = tail_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = slots_[tail & kMask];
            // acquire: synchronizes with the release store from a prior consumer's pop,
            // ensuring we see the completed consumption before reusing the slot
            size_t seq = slot.sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(tail);
            if (diff == 0) {
                // Slot is ready for writing. Try to claim this position.
                // relaxed: the actual publish happens via the sequence store below
                if (tail_.compare_exchange_weak(tail, tail + 1, std::memory_order_relaxed)) {
                    slot.data = std::move(val);
                    // release: makes the written data visible to the consumer that
                    // will acquire this sequence value
                    slot.sequence.store(tail + 1, std::memory_order_release);
                    return true;
                }
                // CAS failed — tail reloaded by compare_exchange_weak, retry
            } else if (diff < 0) {
                // Slot still occupied by unconsumed data — queue is full
                return false;
            } else {
                // Another producer claimed this slot; reload tail and retry
                // relaxed: just re-reading the cursor, no data dependency
                tail = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    // Returns false if queue is empty.
    bool pop(T& val) {
        // relaxed: we will validate via the slot's sequence
        size_t head = head_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = slots_[head & kMask];
            // acquire: synchronizes with the release store from the producer's push,
            // ensuring we see the data written into slot.data
            size_t seq = slot.sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(head + 1);
            if (diff == 0) {
                // Slot contains data at our expected sequence. Claim it.
                // relaxed: the actual "I'm done with this slot" signal is the
                // sequence store below
                if (head_.compare_exchange_weak(head, head + 1, std::memory_order_relaxed)) {
                    val = std::move(slot.data);
                    // release: signals to a future producer that this slot is free
                    // for reuse (the producer will acquire this value)
                    slot.sequence.store(head + Capacity, std::memory_order_release);
                    return true;
                }
                // CAS failed — head reloaded, retry
            } else if (diff < 0) {
                // Sequence behind expected — slot hasn't been written yet; queue empty
                return false;
            } else {
                // Another consumer took this slot; reload head and retry
                // relaxed: no ordering needed, just rereading the cursor
                head = head_.load(std::memory_order_relaxed);
            }
        }
    }

    // Approximate size — not linearizable, useful for monitoring only.
    size_t size_approx() const {
        // relaxed: this is a best-effort metric, not used for synchronization
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t h = head_.load(std::memory_order_relaxed);
        return t >= h ? t - h : 0;
    }
};
