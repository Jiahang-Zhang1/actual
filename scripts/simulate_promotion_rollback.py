#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def write_model_dir(path: Path, top3: float, macro_f1: float, version: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(
        json.dumps({"top3_accuracy": top3, "macro_f1": macro_f1}, indent=2),
        encoding="utf-8",
    )
    (path / "metadata.json").write_text(
        json.dumps({"model_version": version}, indent=2),
        encoding="utf-8",
    )
    (path / "model.joblib").write_text("placeholder model artifact for rollout simulation\n", encoding="utf-8")


def run(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate model promotion and rollback without retraining dependencies.")
    parser.add_argument("--workspace", default="artifacts/test-rollout")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    workspace = repo / args.workspace
    shutil.rmtree(workspace, ignore_errors=True)

    deployed = workspace / "deployed"
    challenger = workspace / "challenger"
    archive = workspace / "archive"
    write_model_dir(deployed, top3=0.72, macro_f1=0.56, version="champion-v1")
    write_model_dir(challenger, top3=0.86, macro_f1=0.68, version="challenger-v2")

    run(
        [
            sys.executable,
            "scripts/promote_model.py",
            "--challenger-dir",
            str(challenger),
            "--deployed-dir",
            str(deployed),
            "--archive-dir",
            str(archive),
        ],
        repo,
    )
    promoted_version = json.loads((deployed / "metadata.json").read_text(encoding="utf-8"))["model_version"]

    run(
        [
            sys.executable,
            "scripts/rollback_model.py",
            "--archive-dir",
            str(archive),
            "--deployed-dir",
            str(deployed),
        ],
        repo,
    )
    rolled_back_version = json.loads((deployed / "metadata.json").read_text(encoding="utf-8"))["model_version"]
    print(json.dumps({"promoted_version": promoted_version, "rolled_back_version": rolled_back_version}, indent=2))


if __name__ == "__main__":
    main()
