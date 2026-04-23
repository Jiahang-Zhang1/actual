import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply quality gates to evaluation artifacts before promotion."
    )
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--per-class-metrics-file", required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--min-top1-accuracy", type=float, default=0.60)
    parser.add_argument("--min-top3-accuracy", type=float, default=0.80)
    parser.add_argument("--min-macro-f1", type=float, default=0.50)
    parser.add_argument("--min-high-confidence-precision", type=float, default=0.75)
    parser.add_argument("--min-mapped-top1-accuracy", type=float, default=0.60)
    parser.add_argument("--per-class-recall-min", type=float, default=0.30)
    parser.add_argument("--per-class-support-min", type=int, default=10)
    parser.add_argument("--output-file")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    metrics_blob = load_json(args.metrics_file)
    per_class_blob = load_json(args.per_class_metrics_file)

    split_metrics = metrics_blob.get(args.split, metrics_blob)
    per_class_metrics = per_class_blob.get(args.split, {})
    failures: list[str] = []

    checks = [
        ("top1_accuracy", args.min_top1_accuracy),
        ("top3_accuracy", args.min_top3_accuracy),
        ("macro_f1", args.min_macro_f1),
        ("high_confidence_precision", args.min_high_confidence_precision),
        ("mapped_top1_accuracy", args.min_mapped_top1_accuracy),
    ]
    for metric_name, threshold in checks:
        metric_value = float(split_metrics.get(metric_name, 0.0))
        if metric_value < threshold:
            failures.append(
                f"{metric_name}={metric_value:.4f} below threshold {threshold:.4f}"
            )

    for label, label_metrics in sorted(per_class_metrics.items()):
        support = int(label_metrics.get("support", 0))
        recall = float(label_metrics.get("recall", 0.0))
        if support >= args.per_class_support_min and recall < args.per_class_recall_min:
            failures.append(
                f"class '{label}' recall={recall:.4f} below threshold "
                f"{args.per_class_recall_min:.4f} with support={support}"
            )

    result = {
        "passed": len(failures) == 0,
        "split": args.split,
        "thresholds": {
            "min_top1_accuracy": args.min_top1_accuracy,
            "min_top3_accuracy": args.min_top3_accuracy,
            "min_macro_f1": args.min_macro_f1,
            "min_high_confidence_precision": args.min_high_confidence_precision,
            "min_mapped_top1_accuracy": args.min_mapped_top1_accuracy,
            "per_class_recall_min": args.per_class_recall_min,
            "per_class_support_min": args.per_class_support_min,
        },
        "metrics": split_metrics,
        "failures": failures,
    }

    output_path = Path(args.output_file) if args.output_file else Path(args.metrics_file).with_name("gate_result.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    if args.fail_on_error and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
