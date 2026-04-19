from __future__ import annotations

import argparse
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from tools.common import percentile, read_json, write_json


_thread_local = threading.local()


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop arrival pattern benchmark")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--endpoint", default="/predict")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--distribution", choices=["constant", "poisson"], default="poisson")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def version_info(base_url: str) -> dict:
    response = requests.get(f"{base_url.rstrip('/')}/versionz", timeout=5)
    response.raise_for_status()
    return response.json()


def send_one(base_url: str, endpoint: str, payload: dict, timeout: float) -> dict:
    session = get_session()
    start = time.perf_counter()
    try:
        response = session.post(f"{base_url.rstrip('/')}{endpoint}", json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": 200 <= response.status_code < 300, "status_code": response.status_code, "latency_ms": latency_ms}
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"ok": False, "status_code": 0, "latency_ms": latency_ms}


def main() -> None:
    args = parse_args()
    payload = read_json(args.request_json)

    futures = []
    start = time.perf_counter()
    next_send = start
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        while True:
            now = time.perf_counter()
            if now - start >= args.duration:
                break

            if now < next_send:
                time.sleep(next_send - now)

            futures.append(executor.submit(send_one, args.base_url, args.endpoint, payload, args.timeout))
            interval = (
                1.0 / args.request_rate
                if args.distribution == "constant"
                else random.expovariate(args.request_rate)
            )
            next_send = max(next_send + interval, time.perf_counter())

        records = [future.result() for future in as_completed(futures)]

    wall_s = max(time.perf_counter() - start, 1e-9)
    latencies = [record["latency_ms"] for record in records]
    ok_count = sum(1 for record in records if record["ok"])
    version = version_info(args.base_url)

    summary = {
        "endpoint_url": f"{args.base_url.rstrip('/')}{args.endpoint}",
        "backend_kind": version["backend_kind"],
        "model_version": version["model_version"],
        "code_version": version["code_version"],
        "model_path": version["model_path"],
        "source_model_path": version["source_model_path"],
        "providers": version["providers"],
        "hardware": version["hardware"],
        "request_rate_target": args.request_rate,
        "distribution": args.distribution,
        "requests_sent": len(records),
        "p50_latency_ms": round(percentile(latencies, 50), 4),
        "p95_latency_ms": round(percentile(latencies, 95), 4),
        "p99_latency_ms": round(percentile(latencies, 99), 4),
        "throughput_rps": round(len(records) / wall_s, 4),
        "error_rate": round(1.0 - (ok_count / len(records) if records else 0.0), 6),
    }
    write_json(Path(args.output_json), summary)
    print(Path(args.output_json).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
