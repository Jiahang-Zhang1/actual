import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.taxonomy import canonicalize_category, taxonomy_manifest


VALID_FEEDBACK_STATUSES = ("accepted_top1", "accepted_top3", "overridden")


def load_feedback_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    query = f"""
    WITH latest_feedback AS (
      SELECT f.*
      FROM ml_feedback f
      JOIN (
        SELECT transaction_id, MAX(created_at) AS max_created_at
        FROM ml_feedback
        GROUP BY transaction_id
      ) latest
        ON latest.transaction_id = f.transaction_id
       AND latest.max_created_at = f.created_at
    )
    SELECT
      lf.transaction_id,
      COALESCE(c.name, lf.final_category_id) AS category,
      lf.final_category_id AS final_category_id,
      lf.feedback_status,
      lf.model_version,
      lf.created_at,
      t.date AS transaction_date,
      t.amount,
      t.account AS account_id,
      COALESCE(t.currency, 'unknown') AS currency,
      'unknown' AS country,
      COALESCE(t.imported_description, p.name, t.notes, '') AS transaction_description,
      t.notes,
      t.imported_description
    FROM latest_feedback lf
    JOIN transactions t
      ON t.id = lf.transaction_id
    LEFT JOIN payees p
      ON p.id = t.payee
    LEFT JOIN categories c
      ON c.id = lf.final_category_id
    WHERE lf.feedback_status IN ({",".join(["?"] * len(VALID_FEEDBACK_STATUSES))})
    """
    return pd.read_sql_query(query, conn, params=list(VALID_FEEDBACK_STATUSES))


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["transaction_description"] = (
        df["transaction_description"].fillna("").astype(str).str.strip()
    )
    df["category_raw"] = df["category"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].map(canonicalize_category)

    df["event_time"] = pd.to_datetime(df["created_at"], errors="coerce")
    fallback_time = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["event_time"] = df["event_time"].fillna(fallback_time)

    unsupported_mask = df["category"].isna()
    dropped_rows = int(unsupported_mask.sum())
    dropped_labels = sorted(
        value
        for value in df.loc[unsupported_mask, "category_raw"].dropna().unique().tolist()
        if value
    )
    df = df.loc[~unsupported_mask].copy()

    df = df.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)

    # Keep the latest feedback per transaction after sorting to avoid duplicates.
    df = df.drop_duplicates(subset=["transaction_id"], keep="last").reset_index(drop=True)
    df.attrs["dropped_unsupported_rows"] = dropped_rows
    df.attrs["dropped_unsupported_labels"] = dropped_labels
    return df


def split_time_ordered(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
):
    n = len(df)
    if n == 0:
        empty = df.iloc[0:0].copy()
        return empty, empty, empty

    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio))) if n > 2 else n

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    if val.empty and len(test) > 1:
        val = test.iloc[:1].copy()
        test = test.iloc[1:].copy()

    return train, val, test


def save_split(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Build versioned train/val/test datasets from Actual feedback data."
    )
    parser.add_argument("--db-path", required=True, help="Path to the Actual SQLite database.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where train/val/test datasets and manifest.json will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--file-format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output format for the split datasets.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    conn = sqlite3.connect(args.db_path)
    try:
        df = load_feedback_dataframe(conn)
    finally:
        conn.close()

    df = prepare_dataframe(df)
    train_df, val_df, test_df = split_time_ordered(df, args.train_ratio, args.val_ratio)

    suffix = ".csv" if args.file_format == "csv" else ".parquet"
    save_split(train_df, output_dir / f"train{suffix}")
    save_split(val_df, output_dir / f"val{suffix}")
    save_split(test_df, output_dir / f"test{suffix}")

    manifest = {
        "source_db_path": args.db_path,
        "row_count": len(df),
        "taxonomy_version": taxonomy_manifest()["taxonomy_version"],
        "dropped_unsupported_rows": int(df.attrs.get("dropped_unsupported_rows", 0)),
        "dropped_unsupported_labels": df.attrs.get("dropped_unsupported_labels", []),
        "category_distribution": {
            str(label): int(count)
            for label, count in df["category"].value_counts().sort_index().items()
        }
        if "category" in df.columns
        else {},
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "file_format": args.file_format,
        "columns": list(df.columns),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
