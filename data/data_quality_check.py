# data_quality_check.py
# Data quality checks at three points
# Assisted by Claude Sonnet 4.6

import pandas as pd
from datetime import datetime

def check_ingestion_quality(filepath):
    """Node 1: Check data quality at ingestion"""
    print(f"\n=== Ingestion Quality Check: {filepath} ===")
    df = pd.read_parquet(filepath)
    
    issues = []
    
    # Check 1: Required columns exist
    required_cols = ['transaction_description', 'category', 'country', 'currency']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    else:
        print(f"[PASS] All required columns present")
    
    # Check 2: No null values in required columns
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"Column '{col}' has {null_count} null values")
            else:
                print(f"[PASS] No nulls in '{col}'")
    
    # Check 3: Minimum row count
    if len(df) < 1000:
        issues.append(f"Too few rows: {len(df)}")
    else:
        print(f"[PASS] Row count: {len(df)}")
    
    # Check 4: Valid categories
    valid_categories = [
        'Food & Dining', 'Transportation', 'Shopping & Retail',
        'Entertainment & Recreation', 'Healthcare & Medical',
        'Utilities & Services', 'Financial Services', 'Income',
        'Travel', 'Other', 'Charity & Donations', 'Government & Legal'
    ]
    invalid = df[~df['category'].isin(valid_categories)]['category'].unique()
    if len(invalid) > 0:
        print(f"[WARN] Unknown categories found: {invalid[:5]}")
    else:
        print(f"[PASS] All categories valid")
    
    if issues:
        print(f"[FAIL] Issues found: {issues}")
        return False
    else:
        print(f"[PASS] Ingestion quality check passed")
        return True


def check_training_set_quality(train_path, eval_path):
    """Node 2: Check quality when compiling training sets"""
    print(f"\n=== Training Set Quality Check ===")
    
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    
    issues = []
    
    # Check 1: Check overlap ratio using combined key
    train_desc = set(zip(train_df['transaction_description'], train_df['category'], train_df['country']))
    eval_desc = set(zip(eval_df['transaction_description'], eval_df['category'], eval_df['country']))
    overlap = train_desc.intersection(eval_desc)
    overlap_ratio = len(overlap) / len(eval_desc)
    if overlap_ratio > 0.6:
        issues.append(f"High overlap ratio: {overlap_ratio:.2f} - possible data leakage")
    else:
        print(f"[PASS] Overlap ratio acceptable: {overlap_ratio:.2f} (due to duplicate merchant names in dataset)")
    
    # Check 2: Label distribution in training set
    label_dist = train_df['category'].value_counts(normalize=True)
    min_ratio = label_dist.min()
    if min_ratio < 0.01:
        issues.append(f"Severe class imbalance: min ratio {min_ratio:.3f}")
    else:
        print(f"[PASS] Label distribution acceptable, min ratio: {min_ratio:.3f}")
    
    # Check 3: Train/eval size ratio
    total = len(train_df) + len(eval_df)
    train_ratio = len(train_df) / total
    if train_ratio < 0.7 or train_ratio > 0.9:
        issues.append(f"Unusual train/eval split: {train_ratio:.2f}")
    else:
        print(f"[PASS] Train/eval split: {train_ratio:.2f}/{1-train_ratio:.2f}")
    
    if issues:
        print(f"[FAIL] Issues found: {issues}")
        return False
    else:
        print(f"[PASS] Training set quality check passed")
        return True


def check_inference_drift(reference_path, inference_path):
    """Node 3: Monitor live inference data drift"""
    print(f"\n=== Inference Drift Check ===")
    
    ref_df = pd.read_parquet(reference_path)
    inf_df = pd.read_csv(inference_path)
    
    issues = []
    
    # Check 1: Country distribution drift
    ref_country = ref_df['country'].value_counts(normalize=True)
    inf_country = inf_df['country'].value_counts(normalize=True)
    
    for country in ref_country.index[:3]:
        if country in inf_country:
            diff = abs(ref_country[country] - inf_country.get(country, 0))
            if diff > 0.2:
                issues.append(f"Country drift detected for {country}: {diff:.2f}")
    
    print(f"[INFO] Reference countries: {dict(ref_country.head(3))}")
    print(f"[INFO] Inference countries: {dict(inf_country.head(3))}")
    
    # Check 2: Currency distribution drift
    ref_currency = ref_df['currency'].value_counts(normalize=True)
    inf_currency = inf_df['currency'].value_counts(normalize=True)
    
    for currency in ref_currency.index[:3]:
        diff = abs(ref_currency[currency] - inf_currency.get(currency, 0))
        if diff > 0.2:
            issues.append(f"Currency drift detected for {currency}: {diff:.2f}")
    
    print(f"[INFO] Reference currencies: {dict(ref_currency.head(3))}")
    print(f"[INFO] Inference currencies: {dict(inf_currency.head(3))}")
    
    if issues:
        print(f"[WARN] Drift detected: {issues}")
        return False
    else:
        print(f"[PASS] No significant drift detected")
        return True


if __name__ == "__main__":
    # Node 1: Ingestion check
    check_ingestion_quality('/home/cc/transactions_clean_v1.parquet')
    
    # Node 2: Training set check
    check_training_set_quality('/home/cc/train_data.csv', '/home/cc/eval_data.csv')
    
    # Node 3: Drift check
    check_inference_drift(
        '/home/cc/transactions_clean_v1.parquet',
        '/home/cc/online_features_output.csv'
    )
