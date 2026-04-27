#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.taxonomy import CANONICAL_CATEGORIES


@dataclass(frozen=True)
class MerchantTemplate:
    payee: str
    category: str
    base_amount: float
    notes: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    def __iter__(self):
        # Backward-compatible tuple unpacking for sparse matrix generators.
        yield self.payee
        yield self.category
        yield self.base_amount


MERCHANTS = [
    MerchantTemplate(
        "Starbucks Store 1458 New York NY",
        "Food & Dining",
        -6.45,
        ("coffee", "morning coffee", "before class", "latte"),
        ("STARBUCKS 1458", "SBUX MOBILE ORDER", "STARBUCKS NO NOTES"),
    ),
    MerchantTemplate(
        "Chipotle Mexican Grill",
        "Food & Dining",
        -14.82,
        ("lunch", "burrito", "mobile order"),
        ("CHIPOTLE ONLINE", "CHIPOTLE MEXICAN GRILL", "CHIPOTLE 3342"),
    ),
    MerchantTemplate(
        "Shake Shack",
        "Food & Dining",
        -17.38,
        ("dinner", "burger", "food with friends"),
        ("SHAKE SHACK", "SHAKESHACK APP", "SHAKE SHACK MADISON SQ"),
    ),
    MerchantTemplate(
        "Panera Bread",
        "Food & Dining",
        -12.64,
        ("lunch", "soup", "bakery"),
        ("PANERA BREAD", "PANERA ONLINE ORDER"),
    ),
    MerchantTemplate(
        "Dominos Pizza",
        "Food & Dining",
        -24.91,
        ("pizza", "dinner delivery", "roommates"),
        ("DOMINOS PIZZA", "DOMINO'S 4321", "DPZ ONLINE"),
    ),
    MerchantTemplate(
        "Subway Store 4521",
        "Food & Dining",
        -9.70,
        ("sandwich", "lunch", "food"),
        ("SUBWAY STORE 4521", "SUBWAY RESTAURANT", "SUBWAY #4521"),
    ),
    MerchantTemplate(
        "Lyft Ride",
        "Transportation",
        -18.20,
        ("ride home", "rideshare", "airport ride", "late night ride"),
        ("LYFT 18.2 RIDE HOME", "LYFT RIDE", "LYFT TRIP", "LYFT *RIDE"),
    ),
    MerchantTemplate(
        "Uber Trip San Francisco CA",
        "Transportation",
        -22.10,
        ("ride to office", "rideshare", "uber home", "trip"),
        ("UBER TRIP", "UBER *TRIP HELP.UBER.COM", "UBER RIDE"),
    ),
    MerchantTemplate(
        "Metro North Railroad",
        "Transportation",
        -16.50,
        ("train ticket", "commute", "railroad"),
        ("METRO NORTH RAILROAD", "MNR TICKET", "TRAIN TICKET"),
    ),
    MerchantTemplate(
        "MTA Subway Fare",
        "Transportation",
        -2.90,
        ("subway fare", "commute", "metro card"),
        ("MTA NYCT PAYGO", "MTA SUBWAY", "OMNY MTA"),
    ),
    MerchantTemplate(
        "NJ Transit",
        "Transportation",
        -11.75,
        ("bus ticket", "train", "commute"),
        ("NJ TRANSIT", "NJT RAIL", "NJTRANSIT APP"),
    ),
    MerchantTemplate(
        "ParkWhiz Inc",
        "Transportation",
        -31.00,
        ("parking", "garage", "event parking"),
        ("PARKWHIZ INC", "PARK WHIZ", "PARKING GARAGE"),
    ),
    MerchantTemplate(
        "EZPass Toll",
        "Transportation",
        -8.25,
        ("toll", "bridge", "highway"),
        ("E-ZPASS NY", "EZPASS TOLL", "TOLL PAYMENT"),
    ),
    MerchantTemplate(
        "Shell Oil 5744332",
        "Transportation",
        -48.20,
        ("gas", "fuel", "car"),
        ("SHELL OIL", "SHELL SERVICE STATION", "KROGER FUEL CENTER"),
    ),
    MerchantTemplate(
        "Amazon Marketplace PMTS",
        "Shopping & Retail",
        -37.53,
        ("online order", "household", "retail"),
        ("AMAZON MKTPLACE PMTS", "AMZN MKTP US", "AMAZON.COM"),
    ),
    MerchantTemplate(
        "Target T-2841 Jersey City",
        "Shopping & Retail",
        -64.19,
        ("groceries and supplies", "household", "store"),
        ("TARGET STORE 0234", "TARGET T-2841", "TARGET.COM"),
    ),
    MerchantTemplate(
        "Costco Whse",
        "Shopping & Retail",
        -132.44,
        ("bulk groceries", "warehouse", "household"),
        ("COSTCO WHSE", "COSTCO WHOLESALE", "COSTCO 312"),
    ),
    MerchantTemplate(
        "Best Buy 00234",
        "Shopping & Retail",
        -84.00,
        ("electronics", "charger", "retail"),
        ("BEST BUY 00234", "BESTBUY.COM", "BEST BUY"),
    ),
    MerchantTemplate(
        "Netflix.com",
        "Entertainment & Recreation",
        -15.49,
        ("subscription", "streaming", "movie night"),
        ("NETFLIX.COM", "NETFLIX", "NETFLIX MONTHLY"),
    ),
    MerchantTemplate(
        "Spotify USA",
        "Entertainment & Recreation",
        -10.99,
        ("music subscription", "streaming", "monthly"),
        ("SPOTIFY USA", "SPOTIFY", "SPOTIFY P2"),
    ),
    MerchantTemplate(
        "AMC Theatres Mobile",
        "Entertainment & Recreation",
        -28.00,
        ("movie", "weekend", "tickets"),
        ("AMC THEATRES", "AMC MOBILE", "AMC LINCOLN SQ"),
    ),
    MerchantTemplate(
        "Verizon Wireless",
        "Utilities & Services",
        -74.10,
        ("phone bill", "monthly bill", "autopay"),
        ("VERIZON WIRELESS", "VZWRLSS", "VERIZON AUTOPAY"),
    ),
    MerchantTemplate(
        "Consolidated Edison",
        "Utilities & Services",
        -118.73,
        ("electric bill", "monthly utility", "autopay"),
        ("CONSOLIDATED EDISON", "CON EDISON", "CONED BILL"),
    ),
    MerchantTemplate(
        "Comcast Cable Autopay",
        "Utilities & Services",
        -91.40,
        ("internet bill", "monthly internet", "service"),
        ("COMCAST CABLE", "XFINITY AUTOPAY", "COMCAST AUTOPAY"),
    ),
    MerchantTemplate(
        "CVS Pharmacy 01452",
        "Healthcare & Medical",
        -18.67,
        ("pharmacy", "medicine", "health"),
        ("CVS PHARMACY 8823", "CVS STORE", "CVS PHARMACY"),
    ),
    MerchantTemplate(
        "Walgreens 6712",
        "Healthcare & Medical",
        -22.40,
        ("pharmacy", "prescription", "health"),
        ("WALGREENS 6712", "WALGREENS", "WAG PHARMACY"),
    ),
    MerchantTemplate(
        "NYC Health",
        "Healthcare & Medical",
        -83.55,
        ("clinic", "medical", "health"),
        ("NYC HEALTH", "NYC HEALTH + HOSP", "CITYMD URGENT CARE"),
    ),
    MerchantTemplate(
        "Payroll Direct Deposit",
        "Income",
        2450.00,
        ("payroll", "salary", "direct deposit"),
        ("PAYROLL DIRECT DEPOSIT", "DIRECT DEP PAYROLL", "SALARY ACH"),
    ),
    MerchantTemplate(
        "ACH Interest Payment",
        "Income",
        4.82,
        ("interest", "bank interest", "savings"),
        ("ACH INTEREST PAYMENT", "INTEREST PAYMENT", "BANK INTEREST"),
    ),
    MerchantTemplate(
        "Bank of America Fee",
        "Financial Services",
        -12.00,
        ("fee", "bank fee", "monthly fee"),
        ("BANK OF AMERICA FEE", "BOA MONTHLY FEE", "ACCOUNT FEE"),
    ),
    MerchantTemplate(
        "Vanguard Investment Transfer",
        "Financial Services",
        -250.00,
        ("investment", "brokerage", "transfer"),
        ("VANGUARD INVESTMENT", "VANGUARD TRANSFER", "BROKERAGE TRANSFER"),
    ),
    MerchantTemplate(
        "Red Cross Donation",
        "Charity & Donations",
        -35.00,
        ("donation", "charity", "nonprofit"),
        ("RED CROSS DONATION", "AMERICAN RED CROSS", "DONATION REDCROSS"),
    ),
    MerchantTemplate(
        "IRS Tax Payment",
        "Government & Legal",
        -410.00,
        ("tax", "government", "irs payment"),
        ("IRS TAX PAYMENT", "US TREASURY TAX", "IRS DIRECT PAY"),
    ),
]

ACCOUNTS = [
    "Bank of America Checking",
    "HSBC Credit Card",
    "Capital One Checking",
    "Ally Savings",
    "transport-card",
    "salary-checking",
]
COUNTRY_CURRENCY = [("US", "USD"), ("GB", "GBP"), ("AU", "AUD")]
GENERIC_NOTES = ("manual entry", "typed by user", "no receipt", "unknown", "")
CITY_SUFFIXES = ("NY", "BROOKLYN NY", "JERSEY CITY NJ", "ONLINE", "APP", "WEB")

assert sorted({merchant.category for merchant in MERCHANTS}) == sorted(CANONICAL_CATEGORIES)

NOTES = {
    category: tuple(
        sorted(
            {
                note
                for merchant in MERCHANTS
                if merchant.category == category
                for note in merchant.notes
            }
        )
    )
    for category in CANONICAL_CATEGORIES
}


def _choice(rng: random.Random, values: tuple[str, ...] | list[str]) -> str:
    return values[rng.randrange(len(values))]


def noisy_payee(template: MerchantTemplate, rng: random.Random) -> str:
    payee = _choice(rng, (template.payee, *template.aliases))
    roll = rng.random()
    if roll < 0.12:
        payee = payee.upper()
    elif roll < 0.20:
        payee = payee.lower()
    elif roll < 0.28:
        payee = payee.title()

    if rng.random() < 0.18:
        payee = f"{payee} {_choice(rng, CITY_SUFFIXES)}"
    if rng.random() < 0.10:
        payee = f"POS {payee}"
    if rng.random() < 0.08:
        payee = payee.replace(" ", "  ")
    if rng.random() < 0.05 and len(payee) > 14:
        payee = payee[: rng.randint(10, len(payee) - 1)]
    return payee


def noisy_notes(template: MerchantTemplate, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.24:
        return ""
    if roll < 0.34:
        return _choice(rng, GENERIC_NOTES)
    if roll < 0.42:
        other = _choice(rng, [item for item in MERCHANTS if item.category != template.category])
        return _choice(rng, other.notes)
    return _choice(rng, template.notes)


def noisy_amount(template: MerchantTemplate, rng: random.Random) -> str:
    amount = template.base_amount
    if amount < 0:
        amount = round(amount * rng.uniform(0.65, 1.55), 2)
    else:
        amount = round(amount * rng.uniform(0.86, 1.14), 2)
    if rng.random() < 0.055:
        amount = -amount
    if rng.random() < 0.035:
        return ""
    return f"{amount:.2f}"


def build_rows(row_count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    today = date.today()
    rows: list[dict[str, str]] = []
    for _ in range(row_count):
        template = _choice(rng, MERCHANTS)
        amount = noisy_amount(template, rng) or f"{template.base_amount:.2f}"
        country, currency = _choice(rng, COUNTRY_CURRENCY)
        rows.append(
            {
                "Date": (today - timedelta(days=rng.randint(0, 150))).isoformat(),
                "Payee": noisy_payee(template, rng),
                "Notes": noisy_notes(template, rng),
                "Amount": amount,
                "Account": _choice(rng, ACCOUNTS),
                "Country": country,
                "Currency": currency,
                "Expected Category": template.category,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Date",
                "Payee",
                "Notes",
                "Amount",
                "Account",
                "Country",
                "Currency",
                "Expected Category",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_qif(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("!Type:Bank\n")
        for row in rows:
            tx_date = date.fromisoformat(row["Date"]).strftime("%m/%d/%Y")
            handle.write(f"D{tx_date}\n")
            handle.write(f"T{row['Amount']}\n")
            handle.write(f"P{row['Payee']}\n")
            if row["Notes"]:
                handle.write(f"M{row['Notes']}\n")
            handle.write("^\n")


def write_serving_batch(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "items": [
            {
                "transaction_description": row["Payee"],
                "country": row.get("Country", "US") or "US",
                "currency": row.get("Currency", "USD") or "USD",
                "amount": float(row["Amount"]),
                "transaction_date": row["Date"],
                "notes": row["Notes"],
            }
            for row in rows
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _training_variant(row: dict[str, str], rng: random.Random) -> dict[str, str]:
    description = row["Payee"]
    notes = row["Notes"]
    amount = row["Amount"]
    imported_description = row["Payee"]

    roll = rng.random()
    if roll < 0.07:
        description = ""
    elif roll < 0.13:
        description = notes
    elif roll < 0.18:
        description = f"{row['Account']} {row['Currency']} amount {amount}"

    if rng.random() < 0.16:
        notes = ""
    if rng.random() < 0.08:
        amount = ""
    if rng.random() < 0.20:
        imported_description = ""

    return {
        "transaction_description": description,
        "category": row["Expected Category"],
        "country": row.get("Country", "US") or "US",
        "currency": row.get("Currency", "USD") or "USD",
        "amount": amount,
        "transaction_date": row["Date"],
        "notes": notes,
        "imported_description": imported_description,
        "account_id": row["Account"],
    }


def write_training_csv(path: Path, rows: list[dict[str, str]], seed: int) -> None:
    rng = random.Random(seed + 17)
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
                "account_id",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(_training_variant(row, rng))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noisy Actual demo transactions for import, serving tests, and training.")
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
    write_training_csv(Path(args.training_output), rows, args.seed)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "csv_output": args.csv_output,
                "qif_output": args.qif_output,
                "batch_output": args.batch_output,
                "training_output": args.training_output,
                "categories": sorted({row["Expected Category"] for row in rows}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
