import pandas as pd
import os
import requests
from sklearn.model_selection import train_test_split
from data_quality_check import check_training_set_quality

DATA_PATH = os.getenv('DATA_PATH', '/home/cc')
SERVING_URL = os.getenv('SERVING_URL', 'http://129.114.25.225:30090')

# Load base dataset
df = pd.read_parquet(f"{DATA_PATH}/transactions_clean_v1.parquet")
df = df.dropna(subset=['transaction_description', 'category'])
df['transaction_description_clean'] = df['transaction_description'].str.lower().str.strip()
df['is_food'] = df['category'] == 'Food & Dining'

# Load real feedback from serving
try:
    response = requests.get(f"{SERVING_URL}/monitor/summary", timeout=5)
    data = response.json()
    selected = data.get('selected_category_counts', {})
    feedback_count = data.get('feedback_count', 0)
    print(f"[INFO] Real feedback from serving: {feedback_count} records")
    print(f"[INFO] User selected categories: {selected}")

    if feedback_count > 0:
        feedback_rows = []
        for category, count in selected.items():
            for _ in range(count):
                feedback_rows.append({
                    'transaction_description': f'user_feedback_{category}',
                    'category': category,
                    'country': 'US',
                    'currency': 'USD',
                    'transaction_description_clean': f'user feedback {category.lower()}',
                    'is_food': category == 'Food & Dining'
                })
        feedback_df = pd.DataFrame(feedback_rows)
        df = pd.concat([df, feedback_df], ignore_index=True)
        print(f"[INFO] Added {len(feedback_df)} feedback records to training data")
except Exception as e:
    print(f"[WARN] Could not load feedback from serving: {e}")

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])
train_df.to_csv(f"{DATA_PATH}/train_data.csv", index=False)
eval_df.to_csv(f"{DATA_PATH}/eval_data.csv", index=False)
print(df.head(5))

check_training_set_quality(f"{DATA_PATH}/train_data.csv", f"{DATA_PATH}/eval_data.csv")
