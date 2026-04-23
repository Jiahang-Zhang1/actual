#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.taxonomy import CANONICAL_CATEGORIES

MERCHANTS = [
    ("Starbucks Store 1458 New York NY", "Food & Dining", -6.45),
    ("Chipotle Online Order", "Food & Dining", -14.82),
    ("Kroger Fuel Center", "Transportation", -48.20),
    ("Uber Trip San Francisco CA", "Transportation", -22.10),
    ("Amazon Marketplace PMTS", "Shopping & Retail", -37.53),
    ("Target T-2841 Jersey City", "Shopping & Retail", -64.19),
    ("Netflix.com", "Entertainment & Recreation", -15.49),
    ("AMC Theatres Mobile", "Entertainment & Recreation", -28.00),
    ("Comcast Cable Autopay", "Utilities & Services", -91.40),
    ("Con Edison Electric", "Utilities & Services", -118.73),
    ("CVS Pharmacy 01452", "Healthcare & Medical", -18.67),
    ("Quest Diagnostics", "Healthcare & Medical", -82.11),
    ("Payroll Direct Deposit", "Income", 2450.00),
    ("ACH Interest Payment", "Income", 4.82),
    ("Bank of America Fee", "Financial Services", -12.00),
    ("Vanguard Investment Transfer", "Financial Services", -250.00),
    ("Red Cross Donation", "Charity & Donations", -35.00),
    ("IRS Tax Payment", "Government & Legal", -410.00),
]

ACCOUNTS = ["Bank of America", "HSBC", "Capital One Checking", "Ally Savings"]
NOTES = {
    "Food & Dining": ["coffee", "lunch", "restaurant"],
    "Transportation": ["commute", "gas", "rideshare"],
    "Shopping & Retail": ["online order", "household", "retail"],
    "Entertainment & Recreation": ["subscription", "movie", "weekend"],
    "Utilities & Services": ["monthly bill", "autopay", "service"],
    "Healthcare & Medical": ["pharmacy", "medical", "health"],
    "Income": ["payroll", "deposit", "interest"],
    "Financial Services": ["fee", "transfer", "investment"],
    "Charity & Donations": ["donation", "charity", "nonprofit"],
    "Government & Legal": ["tax", "government", "legal"],
}

assert sorted(NOTES) == sorted(CANONICAL_CATEGORIES)


def build_rows(row_count: int, seed: int) -> list[dict[str, str]]:
    random.seed(seed)
    today = date.today()
    rows: list[dict[str, str]] = []
    for index in range(row_count):
        merchant, category, base_amount = random.choice(MERCHANTS)
        amount = base_amount
        if amount < 0:
            amount = round(amount * random.uniform(0.75, 1.35), 2)
        else:
            amount = round(amount * random.uniform(0.90, 1.10), 2)
        tx_date = today - timedelta(days=random.randint(0, 90))
        rows.append(
            {
                "Date": tx_date.isoformat(),
                "Payee": merchant,
                "Notes": random.choice(NOTES[category]),
                "Amount": f"{amount:.2f}",
                "Account": random.choice(ACCOUNTS),
                "Expected Category": category,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Date", "Payee", "Notes", "Amount", "Account", "Expected Category"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_qif(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("!Type:Bank\n")
        for row in rows:
            # QIF is the most reliable Actual bank-transaction import path for demos because it avoids CSV column mapping.
            tx_date = date.fromisoformat(row["Date"]).strftime("%m/%d/%Y")
            handle.write(f"D{tx_date}\n")
            handle.write(f"T{row['Amount']}\n")
            handle.write(f"P{row['Payee']}\n")
            handle.write(f"M{row['Notes']}\n")
            handle.write("^\n")


def write_serving_batch(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "items": [
            {
                "transaction_description": row["Payee"],
                "country": "US",
                "currency": "USD",
                "amount": float(row["Amount"]),
                "transaction_date": row["Date"],
                "notes": row["Notes"],
            }
            for row in rows
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_training_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "transaction_description",
                "category",
                "country",
                "currency",
                "amount",
                "transaction_date",
                "notes",
                "imported_description",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "transaction_description": row["Payee"],
                    "category": row["Expected Category"],
                    "country": "US",
                    "currency": "USD",
                    "amount": row["Amount"],
                    "transaction_date": row["Date"],
                    "notes": row["Notes"],
                    "imported_description": row["Payee"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic bank transactions for Actual import and serving tests.")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=9183)
    parser.add_argument("--csv-output", default="artifacts/test-data/actual_bank_transactions.csv")
    parser.add_argument("--qif-output", default="artifacts/test-data/actual_bank_transactions.qif")
    parser.add_argument("--batch-output", default="serving/runtime/batch_input_realistic.json")
    parser.add_argument("--training-output", default="artifacts/test-data/synthetic_training_transactions.csv")
    args = parser.parse_args()

    rows = build_rows(args.rows, args.seed)
    write_csv(Path(args.csv_output), rows)
    write_qif(Path(args.qif_output), rows)
    write_serving_batch(Path(args.batch_output), rows)
    write_training_csv(Path(args.training_output), rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "csv_output": args.csv_output,
                "qif_output": args.qif_output,
                "batch_output": args.batch_output,
                "training_output": args.training_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
