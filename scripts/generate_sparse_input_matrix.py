#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_actual_bank_import import ACCOUNTS, MERCHANTS, NOTES
from serving.app.taxonomy import CANONICAL_CATEGORIES

ALL_FIELDS = [
    "transaction_description",
    "transaction_description_clean",
    "merchant_text",
    "imported_description",
    "notes",
    "amount",
    "currency",
    "country",
    "transaction_date",
    "account_id",
]

VARIANT_METADATA = {
    "full": {
        "blank_style": "none",
        "description": "Fully populated request that mirrors a rich imported transaction.",
    },
    "description_only": {
        "blank_style": "omitted",
        "description": "Only the raw description remains, with defaults for country and currency.",
    },
    "description_amount": {
        "blank_style": "omitted",
        "description": "Description plus amount, but no extra supporting text fields.",
    },
    "description_clean_only": {
        "blank_style": "whitespace",
        "description": "The clean description is present while the raw description is whitespace.",
    },
    "merchant_text_only": {
        "blank_style": "empty-string",
        "description": "Merchant text is the only text source available.",
    },
    "notes_only": {
        "blank_style": "omitted",
        "description": "Only notes are present, simulating a manually typed memo.",
    },
    "payee_no_notes": {
        "blank_style": "empty-string",
        "description": "A normal manual transaction with payee and amount but no notes.",
    },
    "payee_no_amount_no_notes": {
        "blank_style": "omitted",
        "description": "Payee is present, while amount and notes are missing.",
    },
    "payee_positive_amount_no_notes": {
        "blank_style": "empty-string",
        "description": "Payee is present with a positive amount and no notes.",
    },
    "payee_negative_amount_no_notes": {
        "blank_style": "empty-string",
        "description": "Payee is present with a negative amount and no notes.",
    },
    "payee_notes_conflict": {
        "blank_style": "none",
        "description": "Payee and notes intentionally point to different category signals.",
    },
    "payee_amount_conflict": {
        "blank_style": "empty-string",
        "description": "Payee text is clear, but amount sign/size points to a different prior.",
    },
    "camel_case_actual_payload": {
        "blank_style": "omitted",
        "description": "Browser/API clients send camelCase field names instead of snake_case.",
    },
    "invalid_amount_string": {
        "blank_style": "invalid",
        "description": "Amount is a user-entered string that cannot be parsed.",
    },
    "currency_amount_string": {
        "blank_style": "formatted-string",
        "description": "Amount is typed with currency formatting and commas.",
    },
    "unknown_extra_fields": {
        "blank_style": "extra-fields",
        "description": "Payload includes unrelated fields from a form or API caller.",
    },
    "imported_only": {
        "blank_style": "omitted",
        "description": "Only imported_description is available from upstream import metadata.",
    },
    "whitespace_description_notes_fallback": {
        "blank_style": "whitespace",
        "description": "Description is whitespace so notes must take over as the text source.",
    },
    "null_description_imported_fallback": {
        "blank_style": "null",
        "description": "Description is explicitly null so imported_description must be used.",
    },
    "whitespace_all_text_amount": {
        "blank_style": "whitespace",
        "description": "Every text field is whitespace while amount and defaults remain.",
    },
    "null_all_text_amount": {
        "blank_style": "null",
        "description": "Every text field is null while amount and defaults remain.",
    },
    "missing_country_defaults": {
        "blank_style": "omitted",
        "description": "Country is missing and serving must fall back to its default.",
    },
    "missing_currency_defaults": {
        "blank_style": "omitted",
        "description": "Currency is missing and serving must fall back to its default.",
    },
    "missing_amount_full_context": {
        "blank_style": "omitted",
        "description": "Amount is absent but the text context is still rich.",
    },
    "missing_date_full_context": {
        "blank_style": "omitted",
        "description": "Transaction date is absent so weekday bucketing must tolerate it.",
    },
    "account_amount_only": {
        "blank_style": "omitted",
        "description": "Only account, amount, and defaults remain, forcing manual-entry fallback.",
    },
    "amount_date_only": {
        "blank_style": "omitted",
        "description": "Only amount and date remain with default country and currency.",
    },
    "currency_only": {
        "blank_style": "omitted",
        "description": "Country and currency only, with no text or amount.",
    },
    "empty_payload": {
        "blank_style": "omitted",
        "description": "Completely empty payload to verify the API still responds predictably.",
    },
}

VARIANT_NAMES = list(VARIANT_METADATA)

CONFLICT_NOTES = {
    "Charity & Donations": "payroll direct deposit",
    "Entertainment & Recreation": "monthly internet bill",
    "Financial Services": "restaurant dinner",
    "Food & Dining": "irs tax payment",
    "Government & Legal": "lyft ride home",
    "Healthcare & Medical": "amazon retail purchase",
    "Income": "movie tickets",
    "Shopping & Retail": "medical pharmacy visit",
    "Transportation": "payroll deposit",
    "Utilities & Services": "coffee before class",
}


def account_slug(value: str) -> str:
    normalized = (
        value.strip()
        .lower()
        .replace("&", " and ")
        .replace("/", " ")
        .replace("-", " ")
    )
    return "-".join(part for part in normalized.split() if part)


def build_balanced_rows(rows_per_category: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    merchants_by_category: dict[str, list[tuple[str, str, float]]] = {
        category: [] for category in CANONICAL_CATEGORIES
    }
    for merchant_name, category, amount in MERCHANTS:
        merchants_by_category[category].append((merchant_name, category, amount))

    rows: list[dict[str, Any]] = []
    for category in CANONICAL_CATEGORIES:
        category_merchants = merchants_by_category[category]
        for row_index in range(rows_per_category):
            merchant_name, _, base_amount = rng.choice(category_merchants)
            multiplier = rng.uniform(0.9, 1.1) if base_amount > 0 else rng.uniform(0.75, 1.25)
            account_name = rng.choice(ACCOUNTS)
            rows.append(
                {
                    "category": category,
                    "payee": merchant_name,
                    "notes": rng.choice(NOTES[category]),
                    "imported_description": merchant_name,
                    "amount": round(base_amount * multiplier, 2),
                    "currency": "USD",
                    "country": "US",
                    "transaction_date": f"2026-04-{(row_index % 28) + 1:02d}",
                    "account_name": account_name,
                    "account_id": account_slug(account_name),
                }
            )
    return rows


def base_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "transaction_description": row["payee"],
        "transaction_description_clean": row["payee"].lower(),
        "merchant_text": row["payee"],
        "imported_description": row["imported_description"],
        "notes": row["notes"],
        "amount": row["amount"],
        "currency": row["currency"],
        "country": row["country"],
        "transaction_date": row["transaction_date"],
        "account_id": row["account_id"],
    }


def cleanup_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in payload.items():
        if value == "__DROP__":
            continue
        cleaned[key] = value
    return cleaned


def pick_fields(payload: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: payload[field] for field in fields if field in payload}


def variant_payload(variant: str, row: dict[str, Any]) -> dict[str, Any]:
    payload = base_payload(row)

    if variant == "full":
        return payload
    if variant == "description_only":
        return pick_fields(payload, ["transaction_description", "country", "currency"])
    if variant == "description_amount":
        return pick_fields(payload, ["transaction_description", "amount", "country", "currency"])
    if variant == "description_clean_only":
        return {
            "transaction_description": "   ",
            "transaction_description_clean": row["payee"].lower(),
            "amount": row["amount"],
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "merchant_text_only":
        return {
            "transaction_description": "",
            "merchant_text": row["payee"],
            "amount": row["amount"],
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "notes_only":
        return pick_fields(payload, ["notes", "amount", "country", "currency"])
    if variant == "payee_no_notes":
        return {
            "transaction_description": row["payee"],
            "notes": "",
            "amount": row["amount"],
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "payee_no_amount_no_notes":
        return pick_fields(payload, ["transaction_description", "country", "currency"])
    if variant == "payee_positive_amount_no_notes":
        return {
            "transaction_description": row["payee"],
            "notes": "",
            "amount": abs(float(row["amount"])),
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "payee_negative_amount_no_notes":
        return {
            "transaction_description": row["payee"],
            "notes": "",
            "amount": -abs(float(row["amount"])),
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "payee_notes_conflict":
        return {
            "transaction_description": row["payee"],
            "notes": CONFLICT_NOTES[row["category"]],
            "amount": row["amount"],
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "payee_amount_conflict":
        conflict_amount = 2400.0
        if row["category"] == "Income":
            conflict_amount = -88.0
        return {
            "transaction_description": row["payee"],
            "notes": "",
            "amount": conflict_amount,
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "camel_case_actual_payload":
        return {
            "transactionDescription": row["payee"],
            "importedDescription": row["imported_description"],
            "transactionAmount": f"${abs(float(row['amount'])):,.2f}",
            "transactionDate": row["transaction_date"],
            "accountId": row["account_id"],
            "currency": row["currency"],
            "country": row["country"],
        }
    if variant == "invalid_amount_string":
        return {
            "transaction_description": row["payee"],
            "notes": "",
            "amount": "not-a-number",
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "currency_amount_string":
        return {
            "transaction_description": row["payee"],
            "amount": f"${abs(float(row['amount'])):,.2f}",
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "unknown_extra_fields":
        return {
            "transaction_description": row["payee"],
            "amount": row["amount"],
            "country": row["country"],
            "currency": row["currency"],
            "form_state": {"dirty": True, "source": "manual"},
            "memo": "unmapped caller field",
        }
    if variant == "imported_only":
        return pick_fields(payload, ["imported_description", "amount", "country", "currency"])
    if variant == "whitespace_description_notes_fallback":
        return {
            "transaction_description": "   ",
            "notes": row["notes"],
            "amount": row["amount"],
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "null_description_imported_fallback":
        return {
            "transaction_description": None,
            "imported_description": row["imported_description"],
            "amount": row["amount"],
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "whitespace_all_text_amount":
        return {
            "transaction_description": " ",
            "transaction_description_clean": " ",
            "merchant_text": "\t",
            "imported_description": "",
            "notes": "   ",
            "amount": row["amount"],
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "null_all_text_amount":
        return {
            "transaction_description": None,
            "transaction_description_clean": None,
            "merchant_text": None,
            "imported_description": None,
            "notes": None,
            "amount": row["amount"],
            "country": row["country"],
            "currency": row["currency"],
        }
    if variant == "missing_country_defaults":
        payload["country"] = "__DROP__"
        return cleanup_payload(payload)
    if variant == "missing_currency_defaults":
        payload["currency"] = "__DROP__"
        return cleanup_payload(payload)
    if variant == "missing_amount_full_context":
        payload["amount"] = "__DROP__"
        return cleanup_payload(payload)
    if variant == "missing_date_full_context":
        payload["transaction_date"] = "__DROP__"
        return cleanup_payload(payload)
    if variant == "account_amount_only":
        return pick_fields(payload, ["account_id", "amount", "country", "currency"])
    if variant == "amount_date_only":
        return pick_fields(payload, ["amount", "transaction_date", "country", "currency"])
    if variant == "currency_only":
        return pick_fields(payload, ["country", "currency"])
    if variant == "empty_payload":
        return {}
    raise ValueError(f"Unsupported variant: {variant}")


def blank_fields(payload: dict[str, Any]) -> list[str]:
    blanks = []
    for key in ALL_FIELDS:
        if key not in payload:
            blanks.append(key)
            continue
        value = payload[key]
        if value is None:
            blanks.append(key)
        elif isinstance(value, str) and not value.strip():
            blanks.append(key)
    return blanks


def build_sparse_matrix(rows_per_category: int, seed: int) -> list[dict[str, Any]]:
    rows = build_balanced_rows(rows_per_category, seed)
    cases: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for variant in VARIANT_NAMES:
            payload = variant_payload(variant, row)
            blanks = blank_fields(payload)
            metadata = VARIANT_METADATA[variant]
            cases.append(
                {
                    "case_id": f"{variant}-{row_index:04d}",
                    "variant": variant,
                    "blank_style": metadata["blank_style"],
                    "variant_description": metadata["description"],
                    "expected_category": row["category"],
                    "payload": payload,
                    "present_fields": sorted(payload.keys()),
                    "blank_fields": blanks,
                    "blank_field_count": len(blanks),
                    "seed_row": {
                        "payee": row["payee"],
                        "notes": row["notes"],
                        "amount": row["amount"],
                        "account_id": row["account_id"],
                    },
                }
            )
    return cases


def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {variant: 0 for variant in VARIANT_NAMES}
    by_blank_style: dict[str, int] = {}
    by_category = {category: 0 for category in CANONICAL_CATEGORIES}
    for case in cases:
        by_variant[case["variant"]] += 1
        by_category[case["expected_category"]] += 1
        blank_style = case["blank_style"]
        by_blank_style[blank_style] = by_blank_style.get(blank_style, 0) + 1
    return {
        "variant_case_counts": by_variant,
        "blank_style_counts": by_blank_style,
        "category_case_counts": by_category,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_manifest(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "variant",
                "blank_style",
                "expected_category",
                "blank_field_count",
                "blank_fields",
                "present_fields",
                "payee",
                "notes",
                "amount",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "variant": case["variant"],
                    "blank_style": case["blank_style"],
                    "expected_category": case["expected_category"],
                    "blank_field_count": case["blank_field_count"],
                    "blank_fields": ",".join(case["blank_fields"]),
                    "present_fields": ",".join(case["present_fields"]),
                    "payee": case["seed_row"]["payee"],
                    "notes": case["seed_row"]["notes"],
                    "amount": case["seed_row"]["amount"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sparse-input robustness test data with missing, blank, null, and defaulted fields.",
    )
    parser.add_argument("--rows-per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9183)
    parser.add_argument(
        "--json-output",
        default="artifacts/test-data/sparse_input_matrix.json",
    )
    parser.add_argument(
        "--csv-output",
        default="artifacts/test-data/sparse_input_matrix_manifest.csv",
    )
    args = parser.parse_args()

    cases = build_sparse_matrix(args.rows_per_category, args.seed)
    payload = {
        "schema_version": "2026-04-23",
        "rows_per_category": args.rows_per_category,
        "field_names": ALL_FIELDS,
        "variant_metadata": VARIANT_METADATA,
        "case_count": len(cases),
        "summary": build_summary(cases),
        "items": cases,
    }
    write_json(Path(args.json_output), payload)
    write_csv_manifest(Path(args.csv_output), cases)
    print(
        json.dumps(
            {
                "json_output": args.json_output,
                "csv_output": args.csv_output,
                "case_count": len(cases),
                "variant_count": len(VARIANT_NAMES),
                "blank_styles": sorted({case["blank_style"] for case in cases}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
