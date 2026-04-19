import pandas as pd
from data_quality_check import check_training_set_quality

df = pd.read_parquet("/home/cc/transactions_clean_v1.parquet")
df = df.dropna(subset=['transaction_description', 'category'])
df['transaction_description_clean'] = df['transaction_description'].str.lower().str.strip()
df['is_food'] = df['category'] == 'Food & Dining'

train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)
train_df.to_csv("/home/cc/train_data.csv", index=False)
eval_df.to_csv("/home/cc/eval_data.csv", index=False)
print(df.head(5))

# Automatic quality check after compiling training sets
check_training_set_quality("/home/cc/train_data.csv", "/home/cc/eval_data.csv")
