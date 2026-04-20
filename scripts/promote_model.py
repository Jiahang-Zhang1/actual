import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def load_metrics(model_dir: Path) -> dict:
    return json.loads((model_dir / "metrics.json").read_text())


def copy_model_dir(source: Path, target: Path):
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def main():
    parser = argparse.ArgumentParser(description="Promote or reject a challenger model.")
    parser.add_argument("--challenger-dir", required=True)
    parser.add_argument("--deployed-dir", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--metric", default="top3_accuracy")
    parser.add_argument("--min-top3-accuracy", type=float, default=0.70)
    parser.add_argument("--min-macro-f1", type=float, default=0.55)
    args = parser.parse_args()

    challenger_dir = Path(args.challenger_dir)
    deployed_dir = Path(args.deployed_dir)
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    challenger_metrics = load_metrics(challenger_dir)
    champion_metrics = (
        load_metrics(deployed_dir)
        if (deployed_dir / "metrics.json").exists()
        else {args.metric: 0.0, "macro_f1": 0.0}
    )

    challenger_ok = (
        challenger_metrics.get("top3_accuracy", 0.0) >= args.min_top3_accuracy
        and challenger_metrics.get("macro_f1", 0.0) >= args.min_macro_f1
        and challenger_metrics.get(args.metric, 0.0) >= champion_metrics.get(args.metric, 0.0)
    )

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": args.metric,
        "challenger_metrics": challenger_metrics,
        "champion_metrics": champion_metrics,
        "promoted": challenger_ok,
    }

    if challenger_ok:
        if deployed_dir.exists():
            archived_target = archive_dir / datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            shutil.copytree(deployed_dir, archived_target)
        copy_model_dir(challenger_dir, deployed_dir)

    (archive_dir / "last_decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
