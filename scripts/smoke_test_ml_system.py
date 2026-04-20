import argparse
import json
import sqlite3
from pathlib import Path

import requests


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def main():
    parser = argparse.ArgumentParser(description="Smoke test the Actual smart transaction ML system.")
    parser.add_argument("--health-url", default="http://localhost:8000/healthz")
    parser.add_argument("--predict-url", default="http://localhost:8000/predict")
    parser.add_argument(
        "--input-json",
        default="contracts/model_input_sample.json",
        help="Path to a JSON payload to POST to /predict.",
    )
    parser.add_argument("--db-path", help="Optional Actual SQLite database path.")
    args = parser.parse_args()

    health = requests.get(args.health_url, timeout=5)
    health.raise_for_status()
    print("health:", json.dumps(health.json(), indent=2))

    payload = json.loads(Path(args.input_json).read_text())
    predict = requests.post(args.predict_url, json=payload, timeout=5)
    predict.raise_for_status()
    print("predict:", json.dumps(predict.json(), indent=2))

    if args.db_path:
        conn = sqlite3.connect(args.db_path)
        try:
            for table in ("ml_predictions", "ml_feedback"):
                exists = table_exists(conn, table)
                print(f"sqlite table {table}: {'present' if exists else 'missing'}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
