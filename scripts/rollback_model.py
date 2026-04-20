import argparse
import json
import os
import shutil
from pathlib import Path


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

    # Keep the MLflow registry aligned with the file-based rollback so the
    # weekly automation has a clear audit trail of which model is production.
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
    parser = argparse.ArgumentParser(description="Rollback to the most recent archived model.")
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--deployed-dir", required=True)
    parser.add_argument(
        "--mlflow-target-alias",
        default=os.environ.get("MLFLOW_PROMOTION_ALIAS", "production"),
        help="MLflow alias to point at the restored model.",
    )
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    deployed_dir = Path(args.deployed_dir)

    candidates = sorted(
        [path for path in archive_dir.iterdir() if path.is_dir()],
        reverse=True,
    )
    if not candidates:
        raise SystemExit("No archived model directories were found.")

    latest = candidates[0]
    metadata = load_metadata(latest)
    if deployed_dir.exists():
        shutil.rmtree(deployed_dir)
    shutil.copytree(latest, deployed_dir)
    result = {
        "rolled_back_to": str(latest),
        "mlflow_alias": set_mlflow_alias(metadata, args.mlflow_target_alias),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
