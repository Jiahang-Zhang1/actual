#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def run(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def ensure_payload(repo: Path, batch_path: Path, training_path: Path, rows: int) -> None:
    if batch_path.exists() and training_path.exists():
        return
    # The simulation uses the same realistic transaction generator as the demo
    # import flow so online traffic and retraining data share one contract.
    run(
        [
            sys.executable,
            "scripts/generate_actual_bank_import.py",
            "--rows",
            str(rows),
            "--batch-output",
            str(batch_path),
            "--training-output",
            str(training_path),
        ],
        repo,
    )


def wait_for_serving(serving_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(f"{serving_url}/readyz", timeout=5)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"Serving did not become ready: {serving_url}")


def post_feedback(
    serving_url: str,
    transaction_id: str,
    response_item: dict[str, Any],
    top1_accept_rate: float,
    top3_accept_rate: float,
) -> None:
    top_categories = response_item.get("top_categories", [])
    candidates = [str(item.get("category_id", "")) for item in top_categories if item.get("category_id")]
    if not candidates:
        return

    draw = random.random()
    if draw < top1_accept_rate:
        applied = candidates[0]
    elif draw < top3_accept_rate and len(candidates) > 1:
        applied = random.choice(candidates[1:])
    else:
        # This represents user correction outside Top-3 and should pressure
        # rollback/promotion gates during degraded model behavior.
        applied = "manual-user-category"

    requests.post(
        f"{serving_url}/feedback",
        json={
            "transaction_id": transaction_id,
            "model_version": response_item.get("model_version", "unknown"),
            "predicted_category_id": response_item.get("predicted_category_id", ""),
            "applied_category_id": applied,
            "confidence": response_item.get("confidence"),
            "candidate_category_ids": candidates,
        },
        timeout=10,
    ).raise_for_status()


def send_hour_of_traffic(
    serving_url: str,
    items: list[dict[str, Any]],
    simulated_hour: int,
    requests_per_hour: int,
    batch_size: int,
    top1_accept_rate: float,
    top3_accept_rate: float,
) -> dict[str, Any]:
    predicted_items = 0
    feedback_events = 0
    for request_index in range(requests_per_hour):
        batch = random.sample(items, k=min(batch_size, len(items)))
        response = requests.post(
            f"{serving_url}/predict_batch",
            json={"items": batch},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        for item_index, response_item in enumerate(payload.get("items", [])):
            predicted_items += 1
            transaction_id = f"week-sim-{simulated_hour}-{request_index}-{item_index}"
            post_feedback(
                serving_url,
                transaction_id,
                response_item,
                top1_accept_rate,
                top3_accept_rate,
            )
            feedback_events += 1
    return {
        "simulated_hour": simulated_hour,
        "requests": requests_per_hour,
        "predicted_items": predicted_items,
        "feedback_events": feedback_events,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one simulated week of online traffic and automatic model replacement checks."
    )
    parser.add_argument("--serving-url", default="http://127.0.0.1:8000")
    parser.add_argument("--simulated-hours", type=int, default=168)
    parser.add_argument(
        "--seconds-per-hour",
        type=float,
        default=1.0,
        help="Use 3600 for real time, or 1.0 for a compressed demo week.",
    )
    parser.add_argument("--requests-per-hour", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--pipeline-every-hours", type=int, default=24)
    parser.add_argument("--rollout-every-hours", type=int, default=6)
    parser.add_argument("--top1-accept-rate", type=float, default=0.62)
    parser.add_argument("--top3-accept-rate", type=float, default=0.86)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--batch-path", default="serving/runtime/week_simulation_batch.json")
    parser.add_argument("--training-path", default="artifacts/test-data/synthetic_training_transactions.csv")
    parser.add_argument("--log-path", default="artifacts/week-simulation/events.jsonl")
    parser.add_argument("--skip-pipeline", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    batch_path = (repo / args.batch_path).resolve()
    training_path = (repo / args.training_path).resolve()
    log_path = (repo / args.log_path).resolve()

    random.seed(9183)
    ensure_payload(repo, batch_path, training_path, args.rows)
    wait_for_serving(args.serving_url, timeout_seconds=120)

    items = json.loads(batch_path.read_text(encoding="utf-8"))["items"]
    for simulated_hour in range(1, args.simulated_hours + 1):
        started_at = datetime.now(timezone.utc).isoformat()
        traffic_result = send_hour_of_traffic(
            args.serving_url,
            items,
            simulated_hour,
            args.requests_per_hour,
            args.batch_size,
            args.top1_accept_rate,
            args.top3_accept_rate,
        )
        event = {"ts": started_at, "type": "traffic", **traffic_result}
        append_jsonl(log_path, event)
        print(json.dumps(event), flush=True)

        if not args.skip_pipeline and simulated_hour % args.pipeline_every_hours == 0:
            run(
                [
                    sys.executable,
                    "scripts/run_mlops_pipeline.py",
                    "--source-training-csv",
                    str(training_path),
                    "--reload-url",
                    f"{args.serving_url}/admin/reload-model",
                ],
                repo,
            )
            append_jsonl(
                log_path,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "type": "pipeline",
                    "simulated_hour": simulated_hour,
                },
            )

        if simulated_hour % args.rollout_every_hours == 0:
            run(
                [
                    sys.executable,
                    "serving/tools/execute_rollout_action.py",
                    "--execute",
                    "--monitor-url",
                    f"{args.serving_url}/monitor/decision",
                    "--reload-url",
                    f"{args.serving_url}/admin/reload-model",
                ],
                repo,
            )
            append_jsonl(
                log_path,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "type": "rollout_decision",
                    "simulated_hour": simulated_hour,
                },
            )

        time.sleep(max(args.seconds_per_hour, 0.0))


if __name__ == "__main__":
    main()
