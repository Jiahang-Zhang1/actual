import argparse
import json
import os
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
    # Latest data-team merchant examples from origin/master.
    "CHIPOTLE MEXICAN GRILL",
    "SHAKE SHACK",
    "PANERA BREAD",
    "DOMINOS PIZZA",
    "SUBWAY STORE 4521",
    "METRO NORTH RAILROAD",
    "LYFT RIDE",
    "PARKWHIZ INC",
    "TARGET STORE 0234",
    "COSTCO WHSE",
    "BEST BUY 00234",
    "VERIZON WIRELESS",
    "CONSOLIDATED EDISON",
    "SPOTIFY USA",
    "WALGREENS 6712",
    "NYC HEALTH",
]
COUNTRIES = ["US", "GB", "AU"]
CURRENCIES = ["USD", "GBP", "AUD"]
ACCOUNTS = ["checking-account", "credit-card", "savings-account"]
NOTES = [
    "coffee before class",
    "monthly internet bill",
    "ride home",
    "manual entry from user",
    "pharmacy purchase",
]


def random_amount() -> float:
    sign = 1 if random.random() < 0.12 else -1
    return sign * round(random.uniform(3, 250), 2)


def full_payload() -> dict:
    description = random.choice(DESCRIPTIONS)
    return {
        "transaction_description": description,
        "country": random.choice(COUNTRIES),
        "currency": random.choice(CURRENCIES),
        "merchant_text": description,
        "amount": random_amount(),
        "transaction_date": time.strftime("%Y-%m-%d"),
    }


SPARSE_VARIANTS = [
    "description_only",
    "notes_only",
    "amount_only",
    "currency_only",
    "account_amount_only",
    "empty_payload",
]


def sparse_payload(variant: str | None = None) -> dict:
    variant = variant or random.choice(SPARSE_VARIANTS)
    if variant == "description_only":
        return {"transaction_description": random.choice(DESCRIPTIONS)}
    if variant == "notes_only":
        return {"notes": random.choice(NOTES)}
    if variant == "amount_only":
        return {"amount": random_amount()}
    if variant == "currency_only":
        return {"currency": random.choice(CURRENCIES)}
    if variant == "account_amount_only":
        return {
            "account_id": random.choice(ACCOUNTS),
            "amount": random_amount(),
            "currency": random.choice(CURRENCIES),
        }
    return {}


def build_payload(sparse_rate: float = 0.25) -> dict:
    if random.random() < sparse_rate:
        return sparse_payload()
    return full_payload()


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * q)
    return ordered[idx]


def worker(
    base_url: str,
    interval_seconds: float,
    timeout: float,
    stop_at: float,
    metrics: dict,
    lock: threading.Lock,
    sparse_rate: float,
    synthetic_header: bool,
):
    session = requests.Session()
    headers = {"X-Actual-Synthetic-Traffic": "true"} if synthetic_header else {}
    while time.time() < stop_at:
        payload = build_payload(sparse_rate=sparse_rate)
        started = time.perf_counter()
        ok = False
        try:
            response = session.post(
                base_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            ok = response.ok
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # pragma: no cover - request failure path
            with lock:
                metrics["errors"] += 1
                metrics["events"].append({"payload": payload, "error": str(exc), "ok": False})
        else:
            latency = time.perf_counter() - started
            with lock:
                metrics["latencies"].append(latency)
                metrics["success"] += 1
                metrics["events"].append(
                    {
                        "payload": payload,
                        "ok": ok,
                        "latency_seconds": latency,
                        "predicted_category_id": body.get(
                            "predicted_category_id",
                        ),
                        "confidence": body.get("confidence"),
                    },
                )
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Generate emulated production traffic for the Actual ML service.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get(
            "SERVING_URL",
            os.environ.get(
                "PRODUCTION_PREDICT_URL",
                "http://129.114.26.122:30090/predict",
            ),
        ),
    )
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument(
        "--rps",
        type=float,
        default=float(os.environ.get("REQUESTS_PER_SECOND", "2")),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1")),
    )
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--sparse-rate", type=float, default=0.25)
    parser.add_argument(
        "--synthetic-header",
        action="store_true",
        help=(
            "Mark requests as synthetic so serving excludes them from "
            "live-user monitor summaries."
        ),
    )
    parser.add_argument("--seed", type=int, help="Optional random seed for reproducible traffic.")
    parser.add_argument("--output-json", help="Optional path to save the emitted request log.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    interval_seconds = max(0.01, args.workers / max(args.rps, 0.1))
    stop_at = time.time() + args.duration_seconds
    metrics = {"success": 0, "errors": 0, "latencies": [], "events": []}
    lock = threading.Lock()

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
                args.sparse_rate,
                args.synthetic_header,
            )

    summary = {
        "url": args.url,
        "duration_seconds": args.duration_seconds,
        "workers": args.workers,
        "requested_rps": args.rps,
        "sparse_rate": args.sparse_rate,
        "synthetic_header": args.synthetic_header,
        "success": metrics["success"],
        "errors": metrics["errors"],
        "p50_latency_ms": round(percentile(metrics["latencies"], 0.50) * 1000, 2),
        "p95_latency_ms": round(percentile(metrics["latencies"], 0.95) * 1000, 2),
        "avg_latency_ms": round(statistics.mean(metrics["latencies"]) * 1000, 2)
        if metrics["latencies"]
        else 0.0,
    }
    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"summary": summary, "events": metrics["events"]}, indent=2),
        )


if __name__ == "__main__":
    main()
