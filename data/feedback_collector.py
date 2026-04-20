import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = os.getenv('ACTUAL_DB_PATH', '/home/cc/actual_budget.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_description TEXT NOT NULL,
            country TEXT,
            currency TEXT,
            predicted_category TEXT,
            corrected_category TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("[PASS] Database initialized")

def store_feedback(transaction_description, country, currency,
                   predicted_category, corrected_category):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ml_feedback
        (transaction_description, country, currency, predicted_category, corrected_category)
        VALUES (?, ?, ?, ?, ?)
    """, (transaction_description, country, currency,
          predicted_category, corrected_category))
    conn.commit()
    conn.close()

def export_feedback(output_path=None):
    if output_path is None:
        output_path = os.getenv('DATA_PATH', '/home/cc') + '/feedback_data.csv'
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM ml_feedback", conn)
    conn.close()
    df.to_csv(output_path, index=False)
    print(f"[PASS] Exported {len(df)} feedback records to {output_path}")
    return df

def simulate_feedback():
    test_cases = [
        ("STARBUCKS STORE 1458", "US", "USD", "Shopping & Retail", "Food & Dining"),
        ("UBER RIDE NYC", "US", "USD", "Food & Dining", "Transportation"),
        ("NETFLIX SUBSCRIPTION", "UK", "GBP", "Utilities & Services", "Entertainment & Recreation"),
        ("AMAZON PURCHASE", "AUSTRALIA", "AUD", "Transportation", "Shopping & Retail"),
        ("CVS PHARMACY", "US", "USD", "Shopping & Retail", "Healthcare & Medical"),
    ]
    for desc, country, currency, predicted, corrected in test_cases:
        store_feedback(desc, country, currency, predicted, corrected)
        print(f"[INFO] Stored feedback: '{desc}' {predicted} -> {corrected}")

if __name__ == "__main__":
    init_db()
    print("\n=== Simulating User Feedback ===")
    simulate_feedback()
    print("\n=== Exporting Feedback Data ===")
    df = export_feedback()
    print(f"[PASS] Feedback loop complete: {len(df)} records ready for retraining")
    print(df.head())
