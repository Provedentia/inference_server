#pragma once

#include <mutex>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>

struct LatencySample {
    uint64_t queue_wait_ns;
    uint64_t batch_wait_ns;
    uint64_t inference_ns;
    uint64_t end_to_end_ns;
};

class Metrics {
public:
    void record(const LatencySample& s);
    void print_stats();            // called by stats thread every second
    std::string metrics_json();    // for /metrics endpoint

private:
    std::mutex                   mu_;
    std::vector<LatencySample>   samples_;
};
