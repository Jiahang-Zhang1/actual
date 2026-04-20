import pandas as pd
import os
from data_quality_check import check_ingestion_quality

DATA_PATH = os.getenv('DATA_PATH', '/home/cc')
input_file = f"{DATA_PATH}/transactions_v1.parquet"
output_file = f"{DATA_PATH}/transactions_clean_v1.parquet"

df = pd.read_parquet(input_file)
df = df.dropna(subset=['transaction_description', 'category'])
df['transaction_description_clean'] = df['transaction_description'].str.lower().str.strip()
df.to_parquet(output_file, index=False)
print(f"Ingested and cleaned data saved to {output_file}")

check_ingestion_quality(output_file)
