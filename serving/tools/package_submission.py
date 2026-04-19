from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tools.common import REPO_ROOT


SUMMARY_COLUMNS = [
    "study",
    "option",
    "endpoint_url",
    "backend_kind",
    "model_version",
    "code_version",
    "hardware",
    "model_path",
    "source_model_path",
    "workers",
    "service_cpus_limit",
    "service_mem_limit",
    "p50_latency_ms",
    "p95_latency_ms",
    "throughput_rps",
    "error_rate",
    "concurrency_tested",
    "batch_size",
    "request_rate_target",
    "arrival_distribution",
    "cpu_pct_avg",
    "cpu_pct_max",
    "memory_mb_avg",
    "memory_mb_max",
    "ready_after_s",
    "providers",
]


def load_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    header = "| " + " | ".join(SUMMARY_COLUMNS) + " |"
    divider = "| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |"
    lines = [header, divider]
    for row in rows:
        values = [str(row.get(column, "")) for column in SUMMARY_COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raw_root = REPO_ROOT / "results/raw"
    summary_root = REPO_ROOT / "results/summary"
    gradescope_root = REPO_ROOT / "artifacts/gradescope"

    rows = load_rows(raw_root)
    if not rows:
        raise RuntimeError("No benchmark rows found under results/raw. Run bench-* first.")

    summary_root.mkdir(parents=True, exist_ok=True)
    gradescope_root.mkdir(parents=True, exist_ok=True)

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            str(row.get("study", "")),
            str(row.get("option", "")),
            float(row.get("p95_latency_ms", 0) or 0),
        ),
    )

    # summary copies
    write_csv(summary_root / "all_results.csv", rows_sorted)
    (summary_root / "all_results.json").write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")
    write_markdown(summary_root / "all_results.md", rows_sorted)

    # gradescope-focused copies
    write_csv(gradescope_root / "serving_options_table.csv", rows_sorted)
    (gradescope_root / "serving_options_table.json").write_text(
        json.dumps(rows_sorted, indent=2), encoding="utf-8"
    )
    write_markdown(gradescope_root / "serving_options_table.md", rows_sorted)

    examples_root = REPO_ROOT / "artifacts/examples"
    for filename in ["input_sample.json", "batch_input_sample.json", "output_sample.json"]:
        source = examples_root / filename
        target = gradescope_root / filename
        target.write_bytes(source.read_bytes())

    container_info = {
        "dockerfile": "docker/Dockerfile",
        "compose_file": "docker-compose.yml",
        "service_name": "serve",
        "tooling_service_name": "tooling",
        "default_endpoint": "http://<HOST>:8000/predict",
    }
    (gradescope_root / "container_info.json").write_text(json.dumps(container_info, indent=2), encoding="utf-8")
    print((gradescope_root / "serving_options_table.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
