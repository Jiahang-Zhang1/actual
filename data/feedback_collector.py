import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


def export_feedback_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT
      f.transaction_id,
      f.model_version,
      f.predicted_category_id,
      f.final_category_id,
      f.feedback_status,
      f.created_at,
      t.date AS transaction_date,
      t.amount,
      t.account AS account_id,
      COALESCE(t.currency, 'unknown') AS currency,
      COALESCE(t.imported_description, p.name, t.notes, '') AS transaction_description
    FROM ml_feedback f
    JOIN transactions t
      ON t.id = f.transaction_id
    LEFT JOIN payees p
      ON p.id = t.payee
    ORDER BY f.created_at DESC
    """
    return pd.read_sql_query(query, conn)


def main():
    parser = argparse.ArgumentParser(description="Export and summarize Actual ML feedback.")
    parser.add_argument("--db-path", required=True, help="Path to the Actual SQLite database.")
    parser.add_argument("--output", required=True, help="Output CSV or parquet file.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        df = export_feedback_dataframe(conn)
    finally:
        conn.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)

    summary = {
        "rows": int(len(df)),
        "feedback_status_counts": df["feedback_status"].value_counts(dropna=False).to_dict()
        if not df.empty
        else {},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
