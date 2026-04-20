import argparse
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


DESCRIPTIONS = [
    "STARBUCKS STORE 1458 NEW YORK NY",
    "UBER TRIP SAN FRANCISCO CA",
    "WHOLE FOODS MARKET BROOKLYN NY",
    "AMAZON MKTPLACE PMTS",
    "NETFLIX.COM",
    "SHELL OIL 5744332",
    "DELTA AIR LINES ATLANTA",
    "CVS PHARMACY 01452",
]
COUNTRIES = ["US", "GB", "AU"]
CURRENCIES = ["USD", "GBP", "AUD"]


def build_payload():
    return {
        "transaction_description": random.choice(DESCRIPTIONS),
        "country": random.choice(COUNTRIES),
        "currency": random.choice(CURRENCIES),
        "amount": round(random.uniform(3, 250), 2),
        "transaction_date": time.strftime("%Y-%m-%d"),
    }


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * q)
    return ordered[idx]


def worker(base_url: str, interval_seconds: float, timeout: float, stop_at: float, metrics: dict, lock: threading.Lock, output_file: Path | None):
    session = requests.Session()
    while time.time() < stop_at:
        payload = build_payload()
        started = time.perf_counter()
        ok = False
        try:
            response = session.post(base_url, json=payload, timeout=timeout)
            ok = response.ok
            response.raise_for_status()
            response.json()
        except Exception as exc:  # pragma: no cover - request failure path
            with lock:
                metrics["errors"] += 1
                metrics["events"].append({"payload": payload, "error": str(exc), "ok": False})
        else:
            latency = time.perf_counter() - started
            with lock:
                metrics["latencies"].append(latency)
                metrics["success"] += 1
                metrics["events"].append({"payload": payload, "ok": ok, "latency_seconds": latency})
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Generate emulated production traffic for the Actual ML service.")
    parser.add_argument("--url", default=os.environ.get("PRODUCTION_PREDICT_URL", "http://localhost:8000/predict"))
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--rps", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--output-json", help="Optional path to save the emitted request log.")
    args = parser.parse_args()

    interval_seconds = max(0.01, args.workers / max(args.rps, 0.1))
    stop_at = time.time() + args.duration_seconds
    metrics = {"success": 0, "errors": 0, "latencies": [], "events": []}
    lock = threading.Lock()
    output_path = Path(args.output_json) if args.output_json else None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _ in range(args.workers):
            executor.submit(
                worker,
                args.url,
                interval_seconds,
                args.timeout,
                stop_at,
                metrics,
                lock,
                output_path,
            )

    summary = {
        "url": args.url,
        "duration_seconds": args.duration_seconds,
        "workers": args.workers,
        "requested_rps": args.rps,
        "success": metrics["success"],
        "errors": metrics["errors"],
        "p50_latency_ms": round(percentile(metrics["latencies"], 0.50) * 1000, 2),
        "p95_latency_ms": round(percentile(metrics["latencies"], 0.95) * 1000, 2),
        "avg_latency_ms": round(statistics.mean(metrics["latencies"]) * 1000, 2)
        if metrics["latencies"]
        else 0.0,
    }
    print(json.dumps(summary, indent=2))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": summary, "events": metrics["events"]}, indent=2))


if __name__ == "__main__":
    import os

    main()
