cmake_minimum_required(VERSION 3.16)
project(nexus CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Main binary ---
add_executable(nexus
    src/main.cpp
    src/acceptor.cpp
    src/thread_pool.cpp
    src/batcher.cpp
    src/metrics.cpp
)
target_include_directories(nexus PRIVATE include)
target_link_libraries(nexus pthread)
target_compile_options(nexus PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g -fsanitize=address,undefined>
)
target_link_options(nexus PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# --- ThreadSanitizer build (separate — incompatible with ASan) ---
add_executable(nexus_tsan
    src/main.cpp
    src/acceptor.cpp
    src/thread_pool.cpp
    src/batcher.cpp
    src/metrics.cpp
)
target_include_directories(nexus_tsan PRIVATE include)
target_compile_options(nexus_tsan PRIVATE -fsanitize=thread -O1 -g)
target_link_options(nexus_tsan PRIVATE -fsanitize=thread)
target_link_libraries(nexus_tsan pthread)

# --- Queue stress test (TSan) ---
add_executable(queue_stress tests/queue_stress.cpp)
target_include_directories(queue_stress PRIVATE include)
target_compile_options(queue_stress PRIVATE -fsanitize=thread -O1 -g)
target_link_options(queue_stress PRIVATE -fsanitize=thread)
target_link_libraries(queue_stress pthread)

# --- Batcher test ---
add_executable(batcher_test tests/batcher_test.cpp src/batcher.cpp src/metrics.cpp)
target_include_directories(batcher_test PRIVATE include)
target_link_libraries(batcher_test pthread)

# --- Queue benchmark (Google Benchmark) ---
find_package(benchmark QUIET)
if(benchmark_FOUND)
    add_executable(queue_benchmark benchmarks/queue_benchmark.cpp)
    target_include_directories(queue_benchmark PRIVATE include)
    target_link_libraries(queue_benchmark benchmark::benchmark pthread)
endif()
