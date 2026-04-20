import pandas as pd
import os
from data_quality_check import check_training_set_quality

DATA_PATH = os.getenv('DATA_PATH', '/home/cc')

df = pd.read_parquet(f"{DATA_PATH}/transactions_clean_v1.parquet")
df = df.dropna(subset=['transaction_description', 'category'])
df['transaction_description_clean'] = df['transaction_description'].str.lower().str.strip()
df['is_food'] = df['category'] == 'Food & Dining'

train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)
train_df.to_csv(f"{DATA_PATH}/train_data.csv", index=False)
eval_df.to_csv(f"{DATA_PATH}/eval_data.csv", index=False)
print(df.head(5))

check_training_set_quality(f"{DATA_PATH}/train_data.csv", f"{DATA_PATH}/eval_data.csv")
