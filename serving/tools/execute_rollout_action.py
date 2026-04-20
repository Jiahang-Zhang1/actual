from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def build_command(action: str, repo_root: Path, args) -> list[str]:
    python_bin = sys.executable
    if action == "promote_candidate":
        return [
            python_bin,
            str(repo_root / "scripts" / "promote_model.py"),
            "--challenger-dir",
            args.challenger_dir,
            "--deployed-dir",
            args.deployed_dir,
            "--archive-dir",
            args.archive_dir,
            "--min-top3-accuracy",
            str(args.min_top3_accuracy),
            "--min-macro-f1",
            str(args.min_macro_f1),
        ]
    if action == "rollback_active":
        return [
            python_bin,
            str(repo_root / "scripts" / "rollback_model.py"),
            "--archive-dir",
            args.archive_dir,
            "--deployed-dir",
            args.deployed_dir,
        ]
    return []


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read serving /monitor/decision and execute promotion/rollback "
            "scripts when a rollout action is recommended."
        )
    )
    parser.add_argument(
        "--monitor-url",
        default="http://localhost:8000/monitor/decision",
        help="Serving monitor decision endpoint.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="HTTP timeout when reading monitor decision.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the recommended action. If omitted, prints dry-run plan only.",
    )
    parser.add_argument(
        "--challenger-dir",
        default="artifacts/challenger",
    )
    parser.add_argument(
        "--deployed-dir",
        default="artifacts/deployed",
    )
    parser.add_argument(
        "--archive-dir",
        default="artifacts/archive",
    )
    parser.add_argument(
        "--min-top3-accuracy",
        type=float,
        default=0.70,
        help="Passed through to promote_model.py",
    )
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=0.55,
        help="Passed through to promote_model.py",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    decision_response = requests.get(args.monitor_url, timeout=args.timeout_seconds)
    decision_response.raise_for_status()
    decision = decision_response.json()

    action = str(decision.get("recommended_action", "hold"))
    command = build_command(action, repo_root, args)

    result: dict[str, Any] = {
        "monitor_url": args.monitor_url,
        "recommended_action": action,
        "execute": args.execute,
        "decision": decision,
        "command": command,
    }

    if not command:
        result["execution"] = {"status": "skipped", "reason": "no action required"}
        print(json.dumps(result, indent=2))
        return

    if not args.execute:
        result["execution"] = {"status": "dry_run", "reason": "pass --execute to run"}
        print(json.dumps(result, indent=2))
        return

    run_result = run_command(command)
    result["execution"] = run_result
    print(json.dumps(result, indent=2))

    if run_result["returncode"] != 0:
        raise SystemExit(run_result["returncode"])


if __name__ == "__main__":
    main()
