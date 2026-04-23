#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_matrix(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def request_headers(traffic_source: str) -> dict[str, str] | None:
    if not traffic_source:
        return None
    return {"X-Actual-Traffic-Source": traffic_source}


def is_valid_prediction(prediction: Any) -> bool:
    if not isinstance(prediction, dict):
        return False
    if not isinstance(prediction.get("predicted_category_id"), str):
        return False
    if not isinstance(prediction.get("confidence"), (int, float)):
        return False
    top_categories = prediction.get("top_categories")
    if not isinstance(top_categories, list) or not top_categories:
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("category_id"), str)
        and isinstance(item.get("score"), (int, float))
        for item in top_categories
    )


def evaluate_prediction(case: dict[str, Any], prediction: dict[str, Any], mode: str) -> dict[str, Any]:
    top_categories = prediction.get("top_categories", [])
    top_ids = [str(item.get("category_id")) for item in top_categories]
    expected = case["expected_category"]
    confidence = float(prediction["confidence"])
    return {
        "case_id": case["case_id"],
        "variant": case["variant"],
        "blank_style": case.get("blank_style"),
        "mode": mode,
        "ok": True,
        "expected_category": expected,
        "predicted_category_id": prediction["predicted_category_id"],
        "confidence": confidence,
        "top1_match": prediction["predicted_category_id"] == expected,
        "top3_match": expected in top_ids,
        "blank_field_count": case["blank_field_count"],
        "blank_fields": case["blank_fields"],
        "present_fields": case.get("present_fields", []),
        "top_categories": top_ids,
    }


def evaluate_failure(
    case: dict[str, Any],
    mode: str,
    message: str,
    *,
    status_code: int | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "variant": case["variant"],
        "blank_style": case.get("blank_style"),
        "mode": mode,
        "ok": False,
        "status_code": status_code,
        "error": message,
        "expected_category": case["expected_category"],
        "blank_field_count": case["blank_field_count"],
        "blank_fields": case["blank_fields"],
        "present_fields": case.get("present_fields", []),
    }


def run_predict(
    base_url: str,
    cases: list[dict[str, Any]],
    timeout: float,
    headers: dict[str, str] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with requests.Session() as session:
        for case in cases:
            try:
                response = session.post(
                    f"{base_url.rstrip('/')}/predict",
                    json=case["payload"],
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if not is_valid_prediction(payload):
                    results.append(evaluate_failure(case, "predict", "invalid predict response shape"))
                    continue
                results.append(evaluate_prediction(case, payload, "predict"))
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                message = exc.response.text if exc.response is not None else str(exc)
                results.append(evaluate_failure(case, "predict", message, status_code=status_code))
            except Exception as exc:
                results.append(evaluate_failure(case, "predict", str(exc)))
    return results


def run_predict_batch(
    base_url: str,
    cases: list[dict[str, Any]],
    timeout: float,
    headers: dict[str, str] | None,
    batch_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with requests.Session() as session:
        for start in range(0, len(cases), batch_size):
            batch_cases = cases[start : start + batch_size]
            try:
                response = session.post(
                    f"{base_url.rstrip('/')}/predict_batch",
                    json={"items": [case["payload"] for case in batch_cases]},
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                items = payload.get("items")
                if not isinstance(items, list) or len(items) != len(batch_cases):
                    message = (
                        "invalid batch response size: "
                        f"expected {len(batch_cases)}, "
                        f"got {len(items) if isinstance(items, list) else 'non-list'}"
                    )
                    for case in batch_cases:
                        results.append(evaluate_failure(case, "predict_batch", message))
                    continue
                for case, item in zip(batch_cases, items):
                    if not is_valid_prediction(item):
                        results.append(evaluate_failure(case, "predict_batch", "invalid predict_batch item shape"))
                        continue
                    results.append(evaluate_prediction(case, item, "predict_batch"))
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                message = exc.response.text if exc.response is not None else str(exc)
                for case in batch_cases:
                    results.append(
                        evaluate_failure(case, "predict_batch", message, status_code=status_code)
                    )
            except Exception as exc:
                for case in batch_cases:
                    results.append(evaluate_failure(case, "predict_batch", str(exc)))
    return results


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["ok"]]
    summary: dict[str, Any] = {
        "total_cases": len(rows),
        "success_count": len(ok_rows),
        "failure_count": len(rows) - len(ok_rows),
        "success_rate": round(len(ok_rows) / len(rows), 4) if rows else 0.0,
        "top1_match_rate": 0.0,
        "top3_match_rate": 0.0,
        "confidence_mean": None,
        "variant_summary": {},
        "sample_failures": [row for row in rows if not row["ok"]][:20],
    }
    if ok_rows:
        summary["top1_match_rate"] = round(
            sum(1 for row in ok_rows if row["top1_match"]) / len(ok_rows),
            4,
        )
        summary["top3_match_rate"] = round(
            sum(1 for row in ok_rows if row["top3_match"]) / len(ok_rows),
            4,
        )
        summary["confidence_mean"] = round(
            statistics.fmean(float(row["confidence"]) for row in ok_rows),
            4,
        )

    grouped_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_variant[row["variant"]].append(row)

    for variant, variant_rows in sorted(grouped_by_variant.items()):
        variant_ok = [row for row in variant_rows if row["ok"]]
        summary["variant_summary"][variant] = {
            "cases": len(variant_rows),
            "success_rate": round(len(variant_ok) / len(variant_rows), 4) if variant_rows else 0.0,
            "top1_match_rate": round(
                sum(1 for row in variant_ok if row["top1_match"]) / len(variant_ok),
                4,
            )
            if variant_ok
            else 0.0,
            "top3_match_rate": round(
                sum(1 for row in variant_ok if row["top3_match"]) / len(variant_ok),
                4,
            )
            if variant_ok
            else 0.0,
            "avg_confidence": round(
                statistics.fmean(float(row["confidence"]) for row in variant_ok),
                4,
            )
            if variant_ok
            else None,
        }
    return summary


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped_by_mode[row["mode"]].append(row)

    by_mode = {mode: summarize_group(rows) for mode, rows in sorted(grouped_by_mode.items())}
    aggregate = summarize_group(results)
    return {
        "aggregate": aggregate,
        "by_mode": by_mode,
    }


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sparse-input robustness tests against /predict and /predict_batch.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--matrix-json",
        default="artifacts/test-data/sparse_input_matrix.json",
    )
    parser.add_argument("--mode", choices=("predict", "predict_batch", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--traffic-source",
        default="sparse-matrix",
        help="Synthetic source label so these requests can be excluded from live monitoring.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/test-results/sparse_input_matrix_summary.json",
    )
    parser.add_argument(
        "--raw-output-json",
        default="artifacts/test-results/sparse_input_matrix_results.json",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Maximum number of HTTP or response-shape failures allowed before exiting non-zero.",
    )
    args = parser.parse_args()

    matrix = read_matrix(args.matrix_json)
    cases = matrix.get("items", [])
    headers = request_headers(args.traffic_source)
    results: list[dict[str, Any]] = []

    if args.mode in {"predict", "both"}:
        results.extend(run_predict(args.base_url, cases, args.timeout, headers))
    if args.mode in {"predict_batch", "both"}:
        results.extend(
            run_predict_batch(
                args.base_url,
                cases,
                args.timeout,
                headers,
                args.batch_size,
            )
        )

    summary = summarize_results(results)
    payload = {
        "base_url": args.base_url,
        "mode": args.mode,
        "traffic_source": args.traffic_source,
        "matrix_case_count": len(cases),
        "matrix_variant_count": len(matrix.get("variant_metadata", {})),
        "result_count": len(results),
        "summary": summary,
    }
    write_json(args.output_json, payload)
    write_json(args.raw_output_json, {"results": results})
    print(json.dumps(payload, indent=2))

    failure_count = int(summary["aggregate"]["failure_count"])
    if failure_count > args.max_failures:
        raise SystemExit(
            f"Sparse-input run exceeded max failures: {failure_count} > {args.max_failures}"
        )


if __name__ == "__main__":
    main()
