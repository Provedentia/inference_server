#include "metrics.hpp"

void Metrics::record(const LatencySample& s) {
    std::lock_guard<std::mutex> lock(mu_);
    samples_.push_back(s);
}

void Metrics::print_stats() {
    std::vector<LatencySample> snap;
    {
        std::lock_guard<std::mutex> lock(mu_);
        snap.swap(samples_);
    }

    if (snap.empty()) return;

    size_t n = snap.size();

    // Sort by end-to-end latency for percentile computation
    std::sort(snap.begin(), snap.end(),
              [](const LatencySample& a, const LatencySample& b) {
                  return a.end_to_end_ns < b.end_to_end_ns;
              });

    auto percentile = [&](double p) -> double {
        size_t idx = static_cast<size_t>(p / 100.0 * (n - 1));
        return static_cast<double>(snap[idx].end_to_end_ns) / 1e6; // ms
    };

    double avg_queue = 0, avg_batch = 0, avg_infer = 0;
    for (auto& s : snap) {
        avg_queue += static_cast<double>(s.queue_wait_ns);
        avg_batch += static_cast<double>(s.batch_wait_ns);
        avg_infer += static_cast<double>(s.inference_ns);
    }
    avg_queue /= n * 1e6;
    avg_batch /= n * 1e6;
    avg_infer /= n * 1e6;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "[stats] reqs=" << n
        << " p50=" << percentile(50) << "ms"
        << " p95=" << percentile(95) << "ms"
        << " p99=" << percentile(99) << "ms"
        << " | avg queue=" << avg_queue << "ms"
        << " batch=" << avg_batch << "ms"
        << " infer=" << avg_infer << "ms\n";
    std::cerr << oss.str();
}

std::string Metrics::metrics_json() {
    std::vector<LatencySample> snap;
    {
        std::lock_guard<std::mutex> lock(mu_);
        snap = samples_; // copy, don't consume
    }

    size_t n = snap.size();
    if (n == 0) {
        return R"({"count":0})";
    }

    std::sort(snap.begin(), snap.end(),
              [](const LatencySample& a, const LatencySample& b) {
                  return a.end_to_end_ns < b.end_to_end_ns;
              });

    auto pct = [&](double p) -> uint64_t {
        size_t idx = static_cast<size_t>(p / 100.0 * (n - 1));
        return snap[idx].end_to_end_ns;
    };

    std::ostringstream oss;
    oss << R"({"count":)" << n
        << R"(,"p50_ns":)" << pct(50)
        << R"(,"p95_ns":)" << pct(95)
        << R"(,"p99_ns":)" << pct(99)
        << "}";
    return oss.str();
}
