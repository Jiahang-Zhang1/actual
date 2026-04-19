# feedback_collector.py
# Collects user feedback and stores for retraining
# Assisted by Claude Sonnet 4.6

import psycopg2
import pandas as pd
import json
from datetime import datetime

DB_CONFIG = {
    'host': 'localhost',
    'database': 'feedback_db',
    'user': 'datauser',
    'password': 'datapass'
}

def init_db():
    """Create feedback table if not exists"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            transaction_description TEXT NOT NULL,
            country VARCHAR(50),
            currency VARCHAR(10),
            predicted_category VARCHAR(100),
            corrected_category VARCHAR(100),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[PASS] Database initialized")

def store_feedback(transaction_description, country, currency, 
                   predicted_category, corrected_category):
    """Store one feedback record"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback 
        (transaction_description, country, currency, predicted_category, corrected_category)
        VALUES (%s, %s, %s, %s, %s)
    """, (transaction_description, country, currency, 
          predicted_category, corrected_category))
    conn.commit()
    cur.close()
    conn.close()

def export_feedback(output_path='/home/cc/feedback_data.csv'):
    """Export feedback data for retraining"""
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT * FROM feedback", conn)
    conn.close()
    df.to_csv(output_path, index=False)
    print(f"[PASS] Exported {len(df)} feedback records to {output_path}")
    return df

def simulate_feedback():
    """Simulate user feedback for testing"""
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
    # Initialize database
    init_db()
    
    # Simulate some user feedback
    print("\n=== Simulating User Feedback ===")
    simulate_feedback()
    
    # Export feedback for retraining
    print("\n=== Exporting Feedback Data ===")
    df = export_feedback()
    print(f"[PASS] Feedback loop complete: {len(df)} records ready for retraining")
    print(df.head())
