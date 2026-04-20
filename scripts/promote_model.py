import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def load_metrics(model_dir: Path) -> dict:
    return json.loads((model_dir / "metrics.json").read_text())


def copy_model_dir(source: Path, target: Path):
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def set_mlflow_alias(metadata: dict, alias: str | None) -> dict:
    model_name = metadata.get("mlflow_model_name")
    model_version = metadata.get("mlflow_model_version")
    if not alias or not model_name or not model_version:
        return {"updated": False, "reason": "missing mlflow model metadata or target alias"}

    # Import lazily so local file-only promotion still works without MLflow
    # dependencies, while the Chameleon automation can update registry aliases.
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        client.set_registered_model_alias(
            name=str(model_name),
            alias=alias,
            version=str(model_version),
        )
        client.set_model_version_tag(
            name=str(model_name),
            version=str(model_version),
            key="actual.role",
            value=alias,
        )
    except Exception as exc:
        return {
            "updated": False,
            "reason": f"failed to update MLflow alias: {exc}",
            "model_name": str(model_name),
            "model_version": str(model_version),
            "alias": alias,
        }
    return {
        "updated": True,
        "model_name": str(model_name),
        "model_version": str(model_version),
        "alias": alias,
    }


def main():
    parser = argparse.ArgumentParser(description="Promote or reject a challenger model.")
    parser.add_argument("--challenger-dir", required=True)
    parser.add_argument("--deployed-dir", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--metric", default="top3_accuracy")
    parser.add_argument("--min-top3-accuracy", type=float, default=0.70)
    parser.add_argument("--min-macro-f1", type=float, default=0.55)
    parser.add_argument(
        "--mlflow-target-alias",
        default=os.environ.get("MLFLOW_PROMOTION_ALIAS", "production"),
        help="MLflow alias to point at the promoted challenger.",
    )
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
    challenger_metadata = load_metadata(challenger_dir)

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
        "challenger_metadata": challenger_metadata,
        "mlflow_alias": {"updated": False, "reason": "challenger not promoted"},
    }

    if challenger_ok:
        if deployed_dir.exists():
            archived_target = archive_dir / datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            shutil.copytree(deployed_dir, archived_target)
        copy_model_dir(challenger_dir, deployed_dir)
        decision["mlflow_alias"] = set_mlflow_alias(
            challenger_metadata,
            args.mlflow_target_alias,
        )

    (archive_dir / "last_decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
