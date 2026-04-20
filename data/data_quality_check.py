import pandas as pd
import argparse
import json
import requests
import sys
import os
from datetime import datetime


def check_ingestion_quality(filepath):
    print(f"\n=== Ingestion Quality Check: {filepath} ===")
    
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    
    issues = []
    results = {}

    required_cols = ['transaction_description', 'category', 'country', 'currency']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    else:
        print(f"[PASS] All required columns present")
    results['required_columns'] = len(missing_cols) == 0

    for col in required_cols:
        if col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                issues.append(f"Column '{col}' has {null_count} null values")
            else:
                print(f"[PASS] No nulls in '{col}'")
    results['null_counts'] = {col: int(df[col].isnull().sum()) for col in required_cols if col in df.columns}

    if len(df) < 1000:
        issues.append(f"Too few rows: {len(df)}")
    else:
        print(f"[PASS] Row count: {len(df)}")
    results['row_count'] = len(df)

    valid_categories = [
        'Food & Dining', 'Transportation', 'Shopping & Retail',
        'Entertainment & Recreation', 'Healthcare & Medical',
        'Utilities & Services', 'Financial Services', 'Income',
        'Travel', 'Other', 'Charity & Donations', 'Government & Legal'
    ]
    if 'category' in df.columns:
        invalid = df[~df['category'].isin(valid_categories)]['category'].unique()
        if len(invalid) > 0:
            print(f"[WARN] Unknown categories found: {list(invalid[:5])}")
        else:
            print(f"[PASS] All categories valid")
        results['unknown_categories'] = list(invalid[:5])

    passed = len(issues) == 0
    if issues:
        print(f"[FAIL] Issues found: {issues}")
    else:
        print(f"[PASS] Ingestion quality check passed")

    results['passed'] = passed
    results['issues'] = issues
    results['check_type'] = 'ingestion'
    results['timestamp'] = datetime.utcnow().isoformat()
    return results


def check_training_set_quality(train_path, eval_path=None):
    print(f"\n=== Training Set Quality Check ===")

    if train_path.endswith('.parquet'):
        train_df = pd.read_parquet(train_path)
    else:
        train_df = pd.read_csv(train_path)

    issues = []
    results = {}

    if eval_path:
        if eval_path.endswith('.parquet'):
            eval_df = pd.read_parquet(eval_path)
        else:
            eval_df = pd.read_csv(eval_path)

        train_desc = set(zip(train_df['transaction_description'], train_df['category'], train_df['country']))
        eval_desc = set(zip(eval_df['transaction_description'], eval_df['category'], eval_df['country']))
        overlap = train_desc.intersection(eval_desc)
        overlap_ratio = len(overlap) / max(len(eval_desc), 1)
        if overlap_ratio > 0.6:
            issues.append(f"High overlap ratio: {overlap_ratio:.2f} - possible data leakage")
        else:
            print(f"[PASS] Overlap ratio acceptable: {overlap_ratio:.2f}")
        results['overlap_ratio'] = overlap_ratio

    label_dist = train_df['category'].value_counts(normalize=True)
    min_ratio = float(label_dist.min())
    if min_ratio < 0.01:
        issues.append(f"Severe class imbalance: min ratio {min_ratio:.3f}")
    else:
        print(f"[PASS] Label distribution acceptable, min ratio: {min_ratio:.3f}")
    results['min_label_ratio'] = min_ratio

    if eval_path:
        total = len(train_df) + len(eval_df)
        train_ratio = len(train_df) / total
        if train_ratio < 0.7 or train_ratio > 0.9:
            issues.append(f"Unusual train/eval split: {train_ratio:.2f}")
        else:
            print(f"[PASS] Train/eval split: {train_ratio:.2f}/{1-train_ratio:.2f}")
        results['train_ratio'] = train_ratio

    passed = len(issues) == 0
    if issues:
        print(f"[FAIL] Issues found: {issues}")
    else:
        print(f"[PASS] Training set quality check passed")

    results['passed'] = passed
    results['issues'] = issues
    results['check_type'] = 'training_set'
    results['timestamp'] = datetime.utcnow().isoformat()
    return results


def check_inference_drift(reference_path, inference_path):
    print(f"\n=== Inference Drift Check ===")

    ref_df = pd.read_parquet(reference_path)
    if inference_path.endswith('.parquet'):
        inf_df = pd.read_parquet(inference_path)
    else:
        inf_df = pd.read_csv(inference_path)

    issues = []
    results = {}

    ref_country = ref_df['country'].value_counts(normalize=True)
    inf_country = inf_df['country'].value_counts(normalize=True)
    for country in ref_country.index[:3]:
        if country in inf_country:
            diff = abs(float(ref_country[country]) - float(inf_country.get(country, 0)))
            if diff > 0.2:
                issues.append(f"Country drift detected for {country}: {diff:.2f}")
    print(f"[INFO] Reference countries: {dict(ref_country.head(3))}")
    print(f"[INFO] Inference countries: {dict(inf_country.head(3))}")

    ref_currency = ref_df['currency'].value_counts(normalize=True)
    inf_currency = inf_df['currency'].value_counts(normalize=True)
    for currency in ref_currency.index[:3]:
        diff = abs(float(ref_currency[currency]) - float(inf_currency.get(currency, 0)))
        if diff > 0.2:
            issues.append(f"Currency drift detected for {currency}: {diff:.2f}")
    print(f"[INFO] Reference currencies: {dict(ref_currency.head(3))}")
    print(f"[INFO] Inference currencies: {dict(inf_currency.head(3))}")

    passed = len(issues) == 0
    if issues:
        print(f"[WARN] Drift detected: {issues}")
    else:
        print(f"[PASS] No significant drift detected")

    results['passed'] = passed
    results['issues'] = issues
    results['check_type'] = 'online_drift'
    results['timestamp'] = datetime.utcnow().isoformat()
    return results


def post_results(results, post_url):
    try:
        response = requests.post(post_url, json=results, timeout=10)
        print(f"[INFO] Posted results to {post_url}: {response.status_code}")
    except Exception as e:
        print(f"[WARN] Failed to post results: {e}")


def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['ingestion', 'training_set', 'online_drift', 'all'],
                        nargs='?', default='all')
    parser.add_argument('--input', help='Input file path')
    parser.add_argument('--eval-input', help='Eval file path for training_set mode')
    parser.add_argument('--reference', help='Reference file for drift check')
    parser.add_argument('--output-json', help='Output JSON file path')
    parser.add_argument('--post-url', help='URL to POST results to serving')
    args = parser.parse_args()

    DATA_PATH = os.getenv('DATA_PATH', '/home/cc')
    results = None

    if args.mode == 'ingestion':
        input_path = args.input or f"{DATA_PATH}/transactions_clean_v1.parquet"
        results = check_ingestion_quality(input_path)

    elif args.mode == 'training_set':
        input_path = args.input or f"{DATA_PATH}/train_data.csv"
        eval_path = args.eval_input or f"{DATA_PATH}/eval_data.csv"
        results = check_training_set_quality(input_path, eval_path)

    elif args.mode == 'online_drift':
        reference_path = args.reference or f"{DATA_PATH}/transactions_clean_v1.parquet"
        input_path = args.input or f"{DATA_PATH}/online_features_output.csv"
        results = check_inference_drift(reference_path, input_path)

    elif args.mode == 'all':
        check_ingestion_quality(f"{DATA_PATH}/transactions_clean_v1.parquet")
        check_training_set_quality(f"{DATA_PATH}/train_data.csv", f"{DATA_PATH}/eval_data.csv")
        check_inference_drift(f"{DATA_PATH}/transactions_clean_v1.parquet",
                              f"{DATA_PATH}/online_features_output.csv")
        sys.exit(0)

    if results:
        if args.output_json:
            save_results(results, args.output_json)
        if args.post_url:
            post_results(results, args.post_url)
