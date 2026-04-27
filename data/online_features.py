import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def clean_text(value: object | None, max_length: int = 512) -> str:
    return str(value or "").replace("\n", " ").strip()[:max_length]


def choose_description(transaction: dict[str, Any]) -> str:
    for key in [
        "transaction_description",
        "transactionDescription",
        "transaction_description_clean",
        "transactionDescriptionClean",
        "merchant_text",
        "merchantText",
        "imported_description",
        "importedDescription",
        "notes",
    ]:
        cleaned = clean_text(transaction.get(key))
        if cleaned:
            return cleaned
    return ""


def fallback_description(transaction: dict[str, Any]) -> str:
    description = choose_description(transaction)
    if description:
        return description

    amount = transaction.get("amount")
    account_id = (
        transaction.get("account_id")
        or transaction.get("accountId")
        or transaction.get("account")
    )
    currency = clean_text(transaction.get("currency"), max_length=16)
    amount_hint = ""
    if amount is not None and amount != "":
        try:
            amount_hint = f"amount {abs(float(amount))}"
        except (TypeError, ValueError):
            amount_hint = ""
    hints = [
        f"account {clean_text(account_id, max_length=64)}" if account_id else "",
        currency if currency and currency.casefold() != "unknown" else "",
        amount_hint,
    ]
    return " ".join(part for part in hints if part).strip() or "manual entry"


def amount_bucket(value) -> str:
    if value is None or value == "":
        return "unknown"
    try:
        amount = abs(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if amount < 20:
        return "micro"
    if amount < 100:
        return "small"
    if amount < 500:
        return "medium"
    return "large"


def weekday_token(value: str | None) -> str:
    if not value:
        return "unknown"
    try:
        return datetime.fromisoformat(value).strftime("%A").lower()
    except ValueError:
        return "unknown"


def compute_features(transaction: dict[str, Any]) -> dict[str, Any]:
    description = fallback_description(transaction)

    return {
        "transaction_description": description,
        "country": str(transaction.get("country") or "unknown").strip() or "unknown",
        "currency": str(transaction.get("currency") or "unknown").strip() or "unknown",
        "amount": transaction.get("amount"),
        "transaction_date": transaction.get("transaction_date")
        or transaction.get("transactionDate"),
        "account_id": transaction.get("account_id") or transaction.get("accountId"),
        "notes": transaction.get("notes"),
        "imported_description": transaction.get("imported_description")
        or transaction.get("importedDescription"),
        "amount_bucket": amount_bucket(transaction.get("amount")),
        "weekday": weekday_token(
            transaction.get("transaction_date") or transaction.get("transactionDate")
        ),
    }


def format_for_serving(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "transaction_description": features["transaction_description"],
        "country": features["country"],
        "currency": features["currency"],
        "amount": features.get("amount"),
        "transaction_date": features.get("transaction_date"),
        "account_id": features.get("account_id"),
        "notes": features.get("notes"),
        "imported_description": features.get("imported_description"),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute online features for Actual ML serving.")
    parser.add_argument("payload_json", help="JSON payload describing one transaction.")
    parser.add_argument("--output", help="Optional JSON file path to save the serving payload.")
    args = parser.parse_args()

    payload = json.loads(args.payload_json)
    features = compute_features(payload)
    serving_payload = format_for_serving(features)

    print(json.dumps(serving_payload, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serving_payload, indent=2))


if __name__ == "__main__":
    main()
