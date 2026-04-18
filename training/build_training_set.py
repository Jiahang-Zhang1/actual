import argparse
import pandas as pd
import sqlite3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)

    df = pd.read_sql_query("""
        SELECT
          f.transaction_id,
          f.final_category_id AS category,
          f.created_at,
          t.imported_description AS transaction_description,
          'unknown' AS country,
          COALESCE(t.currency, 'unknown') AS currency
        FROM ml_feedback f
        JOIN transactions t
          ON t.id = f.transaction_id
        WHERE f.feedback_status IN ('accepted_top1', 'accepted_top3', 'overridden')
    """, conn)

    df["transaction_description"] = (
        df["transaction_description"].fillna("").astype(str).str.strip()
    )
    df = df[df["transaction_description"] != ""].copy()

    # 时间切分前先按时间排序
    df = df.sort_values("created_at").reset_index(drop=True)

    df.to_parquet(args.output, index=False)
    print(f"saved {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()
