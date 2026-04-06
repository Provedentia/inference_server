#!/usr/bin/env python3
"""
Runs wrk against nexus and flask, collects results, writes JSON.
Usage: python load_test.py --nexus-port 8080 --flask-port 8081
Writes: benchmarks/results/baseline_comparison.json
"""
import subprocess
import json
import argparse
import platform
import os
import re
import sys


def run_wrk(port, threads=8, connections=100, duration=30):
    """Run wrk and return parsed results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lua_script = os.path.join(script_dir, "post.lua")

    cmd = [
        "wrk",
        f"-t{threads}",
        f"-c{connections}",
        f"-d{duration}s",
        "--latency",
        "-s", lua_script,
        f"http://localhost:{port}/infer",
    ]

    print(f"  Running: wrk -t{threads} -c{connections} -d{duration}s http://localhost:{port}/infer")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 30,
        )
        return parse_wrk_output(result.stdout)
    except FileNotFoundError:
        print("  ERROR: wrk not installed. Install with: brew install wrk")
        return None
    except subprocess.TimeoutExpired:
        print("  ERROR: wrk timed out")
        return None


def parse_wrk_output(output):
    """Parse wrk output for key metrics."""
    result = {}

    # Requests/sec
    rps_match = re.search(r"Requests/sec:\s+([\d.]+)", output)
    if rps_match:
        result["rps"] = float(rps_match.group(1))

    # Latency percentiles from --latency output
    lat_matches = re.findall(r"\s+(50%|75%|90%|99%)\s+([\d.]+)(\w+)", output)
    for pct, val, unit in lat_matches:
        multiplier = {"us": 0.001, "ms": 1.0, "s": 1000.0}.get(unit, 1.0)
        key = f"p{pct.replace('%', '')}_ms"
        result[key] = float(val) * multiplier

    # Transfer/sec
    transfer_match = re.search(r"Transfer/sec:\s+([\d.]+)(\w+)", output)
    if transfer_match:
        result["transfer_per_sec"] = f"{transfer_match.group(1)}{transfer_match.group(2)}"

    # Total requests and errors
    req_match = re.search(r"(\d+) requests in", output)
    if req_match:
        result["total_requests"] = int(req_match.group(1))

    err_match = re.search(r"Socket errors:.*?(\d+) connect.*?(\d+) read.*?(\d+) write.*?(\d+) timeout", output)
    if err_match:
        result["errors"] = {
            "connect": int(err_match.group(1)),
            "read": int(err_match.group(2)),
            "write": int(err_match.group(3)),
            "timeout": int(err_match.group(4)),
        }

    if not result:
        result["raw_output"] = output

    return result


def hardware_info():
    return {
        "cpu": platform.processor(),
        "cores": os.cpu_count(),
        "platform": platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser(description="Nexus vs Flask load test")
    parser.add_argument("--nexus-port", type=int, default=8080)
    parser.add_argument("--flask-port", type=int, default=8081)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--connections", type=int, default=100)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--nexus-only", action="store_true", help="Only test nexus")
    args = parser.parse_args()

    results = {"hardware": hardware_info()}

    print("Testing Nexus...")
    nexus_result = run_wrk(args.nexus_port, args.threads, args.connections, args.duration)
    if nexus_result:
        results["nexus"] = nexus_result

    if not args.nexus_only:
        print("Testing Flask...")
        flask_result = run_wrk(args.flask_port, args.threads, args.connections, args.duration)
        if flask_result:
            results["flask"] = flask_result

    os.makedirs("results", exist_ok=True)
    output_path = "results/baseline_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
