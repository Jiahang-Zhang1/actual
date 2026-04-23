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


def tag_mlflow_version(metadata: dict, role: str) -> dict:
    model_name = metadata.get("mlflow_model_name")
    model_version = metadata.get("mlflow_model_version")
    if not model_name or not model_version:
        return {"updated": False, "reason": "missing mlflow model metadata"}
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        client.set_model_version_tag(
            name=str(model_name),
            version=str(model_version),
            key="actual.role",
            value=role,
        )
    except Exception as exc:
        return {
            "updated": False,
            "reason": f"failed to tag MLflow version: {exc}",
            "model_name": str(model_name),
            "model_version": str(model_version),
            "role": role,
        }
    return {
        "updated": True,
        "model_name": str(model_name),
        "model_version": str(model_version),
        "role": role,
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

    if not archive_dir.exists():
        # A fresh deployment may request rollback before the first promotion has
        # archived a champion model. Treat this as a safe no-op for automation.
        print(
            json.dumps(
                {
                    "rolled_back_to": None,
                    "rollback_skipped": True,
                    "reason": f"archive directory does not exist: {archive_dir}",
                    "mlflow_alias": {
                        "updated": False,
                        "reason": "no archived model available",
                    },
                },
                indent=2,
            )
        )
        return

    candidates = sorted(
        [path for path in archive_dir.iterdir() if path.is_dir()],
        reverse=True,
    )
    if not candidates:
        # Keep scheduled rollback jobs idempotent when no prior model is
        # available; monitoring still records why rollback was considered.
        print(
            json.dumps(
                {
                    "rolled_back_to": None,
                    "rollback_skipped": True,
                    "reason": f"no archived model directories were found in: {archive_dir}",
                    "mlflow_alias": {
                        "updated": False,
                        "reason": "no archived model available",
                    },
                },
                indent=2,
            )
        )
        return

    latest = candidates[0]
    current_metadata = load_metadata(deployed_dir)
    metadata = load_metadata(latest)
    if deployed_dir.exists():
        shutil.rmtree(deployed_dir)
    shutil.copytree(latest, deployed_dir)
    result = {
        "rolled_back_to": str(latest),
        "restored_metadata": metadata,
        "replaced_metadata": current_metadata,
        "mlflow_alias": set_mlflow_alias(metadata, args.mlflow_target_alias),
        "production_role_tag": tag_mlflow_version(metadata, "production"),
        "replaced_role_tag": tag_mlflow_version(current_metadata, "replaced"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
