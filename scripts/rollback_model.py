import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Rollback to the most recent archived model.")
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--deployed-dir", required=True)
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
    if deployed_dir.exists():
        shutil.rmtree(deployed_dir)
    shutil.copytree(latest, deployed_dir)
    print(f"Rolled back deployed model to {latest}")


if __name__ == "__main__":
    main()
