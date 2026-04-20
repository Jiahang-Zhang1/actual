import json
import re
import sys
import os
import pandas as pd

def compute_features(transaction: dict) -> dict:
    description = transaction.get("transaction_description", "")
    description_clean = re.sub(r'[^a-z0-9\s]', '', description.lower()).strip()
    return {
        "transaction_description": description,
        "country": transaction.get("country", ""),
        "currency": transaction.get("currency", "")
    }

if __name__ == "__main__":
    input_data = json.loads(sys.argv[1])
    features = compute_features(input_data)
    
    print("Features for serving:")
    print(json.dumps(features, indent=2))
    
    DATA_PATH = os.getenv('DATA_PATH', '/home/cc')
    df = pd.DataFrame([features])
    df.to_csv(f"{DATA_PATH}/online_features_output.csv", index=False)
    
    from data_quality_check import check_inference_drift
    check_inference_drift(
        f"{DATA_PATH}/transactions_clean_v1.parquet",
        f"{DATA_PATH}/online_features_output.csv"
    )
