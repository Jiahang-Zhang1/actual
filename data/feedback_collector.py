import sqlite3
import pandas as pd
import os
import requests
from datetime import datetime

FEEDBACK_URL = os.getenv('FEEDBACK_URL', 'http://129.114.25.225:30090/feedback')
SERVING_URL = os.getenv('SERVING_URL', 'http://129.114.25.225:30090')
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

def fetch_real_feedback():
    try:
        response = requests.get(f"{SERVING_URL}/monitor/summary", timeout=5)
        data = response.json()
        feedback_count = data.get('feedback_count', 0)
        selected = data.get('selected_category_counts', {})
        predicted = data.get('predicted_category_counts', {})
        top1_acceptance = data.get('top1_acceptance', 0)
        print(f"\n=== Real Feedback from Serving ===")
        print(f"[INFO] Total feedback records: {feedback_count}")
        print(f"[INFO] Top1 acceptance rate: {top1_acceptance:.2%}")
        print(f"[INFO] User selected categories: {selected}")
        print(f"[INFO] Model predicted categories: {predicted}")
        return data
    except Exception as e:
        print(f"[WARN] Could not fetch real feedback: {e}")
        return None

def export_feedback(output_path=None):
    if output_path is None:
        output_path = os.getenv('DATA_PATH', '/home/cc') + '/feedback_data.csv'
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM ml_feedback", conn)
    conn.close()
    df.to_csv(output_path, index=False)
    print(f"[PASS] Exported {len(df)} feedback records to {output_path}")
    return df

if __name__ == "__main__":
    init_db()
    print("\n=== Fetching Real Feedback from Serving ===")
    fetch_real_feedback()
    print("\n=== Exporting Feedback Data ===")
    df = export_feedback()
    print(f"[PASS] Feedback loop complete: {len(df)} records ready for retraining")
    print(df.head())
