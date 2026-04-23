from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SERVING_ROOT = Path(__file__).resolve().parents[1]
if str(SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVING_ROOT))

from tools.common import percentile, read_json, write_json


_thread_local = threading.local()


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP benchmark for /predict or /predict_batch")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--endpoint", default="/predict")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--traffic-source",
        default="benchmark",
        help="Synthetic source label for excluding benchmark traffic from online monitoring. Use an empty string to send live traffic.",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_payload(path: str, batch_size: int, endpoint: str) -> tuple[dict, int]:
    payload = read_json(path)
    if endpoint.rstrip("/").endswith("predict_batch"):
        if isinstance(payload, dict) and "items" in payload:
            items = list(payload["items"])
            if items and batch_size > 0:
                if len(items) >= batch_size:
                    items = items[:batch_size]
                else:
                    multiplier = (batch_size + len(items) - 1) // len(items)
                    items = (items * multiplier)[:batch_size]
            return {"items": items}, len(items)
        items = [payload for _ in range(batch_size)]
        return {"items": items}, len(items)

    if batch_size > 1 and "items" not in payload:
        return {"items": [payload for _ in range(batch_size)]}, batch_size
    if isinstance(payload, dict) and "items" in payload:
        return payload, len(payload["items"])
    return payload, 1


def version_info(base_url: str) -> dict:
    response = requests.get(f"{base_url.rstrip('/')}/versionz", timeout=5)
    response.raise_for_status()
    return response.json()


def benchmark_headers(traffic_source: str) -> dict[str, str]:
    if not traffic_source:
        return {}
    return {"X-Actual-Traffic-Source": traffic_source}


def send_one(base_url: str, endpoint: str, payload: dict, timeout: float, headers: dict[str, str]) -> dict:
    session = get_session()
    url = f"{base_url.rstrip('/')}{endpoint}"
    start = time.perf_counter()
    try:
        response = session.post(url, json=payload, timeout=timeout, headers=headers or None)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": False,
            "status_code": 0,
            "latency_ms": latency_ms,
        }


def main() -> None:
    args = parse_args()
    payload, items_per_request = load_payload(args.request_json, args.batch_size, args.endpoint)
    headers = benchmark_headers(args.traffic_source)

    for _ in range(min(8, max(1, args.requests // 10))):
        send_one(args.base_url, args.endpoint, payload, args.timeout, headers)

    records = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(send_one, args.base_url, args.endpoint, payload, args.timeout, headers)
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            records.append(future.result())
    wall_s = max(time.perf_counter() - start, 1e-9)

    latencies = [record["latency_ms"] for record in records]
    ok_count = sum(1 for record in records if record["ok"])
    version = version_info(args.base_url)

    payload = {
        "endpoint_url": f"{args.base_url.rstrip('/')}{args.endpoint}",
        "backend_kind": version["backend_kind"],
        "model_version": version["model_version"],
        "code_version": version["code_version"],
        "model_path": version["model_path"],
        "source_model_path": version["source_model_path"],
        "providers": version["providers"],
        "hardware": version["hardware"],
        "requests_sent": len(records),
        "items_per_request": items_per_request,
        "concurrency_tested": args.concurrency,
        "traffic_source": args.traffic_source,
        "p50_latency_ms": round(percentile(latencies, 50), 4),
        "p95_latency_ms": round(percentile(latencies, 95), 4),
        "p99_latency_ms": round(percentile(latencies, 99), 4),
        "throughput_rps": round(len(records) / wall_s, 4),
        "throughput_items_per_sec": round((len(records) * items_per_request) / wall_s, 4),
        "error_rate": round(1.0 - (ok_count / len(records) if records else 0.0), 6),
    }
    write_json(Path(args.output_json), payload)
    print(Path(args.output_json).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
