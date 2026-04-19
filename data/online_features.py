import json
import re
import sys
import pandas as pd

def compute_features(transaction: dict) -> dict:
    description = transaction.get("transaction_description", "")
    description_clean = re.sub(r'[^a-z0-9\s]', '', description.lower()).strip()
    return {
        "transaction_description": description,
        "country": transaction.get("country", ""),
        "currency": transaction.get("currency", "")
    }

def format_for_serving(features: dict) -> dict:
    return {
        "items": [features]
    }

if __name__ == "__main__":
    input_data = json.loads(sys.argv[1])
    features = compute_features(input_data)
    serving_input = format_for_serving(features)
    
    print("Features for serving:")
    print(json.dumps(serving_input, indent=2))
    
    # Save for drift monitoring
    df = pd.DataFrame([features])
    df.to_csv("/home/cc/online_features_output.csv", index=False)
    
    # Automatic drift check
    from data_quality_check import check_inference_drift
    check_inference_drift(
        "/home/cc/transactions_clean_v1.parquet",
        "/home/cc/online_features_output.csv"
    )
