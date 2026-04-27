#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ManualCase:
    name: str
    payload: dict[str, Any]
    endpoint: str = "/predict"
    expected_top1: str | None = None
    max_confidence: float | None = None


CASES = [
    ManualCase(
        "payee_no_notes_normal_amount",
        {
            "transaction_description": "STARBUCKS STORE 1458",
            "notes": "",
            "amount": -7.50,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Food & Dining",
    ),
    ManualCase(
        "swagger_payee_alias_no_notes",
        {
            "payee": "STARBUCKS NO NOTES",
            "memo": "",
            "payment": "7.50",
            "currency": "USD",
        },
        expected_top1="Food & Dining",
    ),
    ManualCase(
        "swagger_payee_name_alias_conflict_amount",
        {
            "payeeName": "Payroll Direct Deposit",
            "comment": "",
            "payment": "-88.00",
            "currency": "USD",
        },
        expected_top1="Income",
        max_confidence=0.8,
    ),
    ManualCase(
        "payee_no_notes_missing_amount",
        {
            "transaction_description": "LYFT RIDE TEST",
            "notes": "",
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Transportation",
    ),
    ManualCase(
        "lyft_ride_home_amount_regression",
        {
            "transaction_description": "lyft 18.2 ride home",
            "notes": "",
            "amount": -18.20,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Transportation",
        max_confidence=0.8,
    ),
    ManualCase(
        "uber_payment_alias_no_notes",
        {
            "payee": "UBER *TRIP HELP.UBER.COM",
            "memo": "",
            "payment": "23.45",
            "currency": "USD",
        },
        expected_top1="Transportation",
        max_confidence=0.8,
    ),
    ManualCase(
        "mta_notes_only",
        {
            "transaction_description": "",
            "notes": "mta subway fare commute",
            "amount": -2.90,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Transportation",
        max_confidence=0.9,
    ),
    ManualCase(
        "notes_no_payee",
        {
            "transaction_description": "",
            "notes": "monthly internet bill",
            "amount": -85.20,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Utilities & Services",
    ),
    ManualCase(
        "amount_only_negative_small",
        {
            "amount": -18.20,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Food & Dining",
    ),
    ManualCase(
        "amount_only_positive_large",
        {
            "amount": 2500.0,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Income",
        max_confidence=0.8,
    ),
    ManualCase(
        "account_amount_only_transport",
        {
            "account_id": "transport-card",
            "amount": -18.20,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Transportation",
        max_confidence=0.8,
    ),
    ManualCase(
        "payee_income_but_payment_negative",
        {
            "transaction_description": "Payroll Direct Deposit",
            "notes": "",
            "amount": -88.0,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Income",
        max_confidence=0.8,
    ),
    ManualCase(
        "payee_food_but_deposit_positive_large",
        {
            "transaction_description": "STARBUCKS STORE 1458",
            "amount": 2400.0,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Food & Dining",
        max_confidence=0.8,
    ),
    ManualCase(
        "payee_transport_notes_income_conflict",
        {
            "transaction_description": "LYFT RIDE",
            "notes": "payroll deposit",
            "amount": -18.20,
            "currency": "USD",
            "country": "US",
        },
        expected_top1="Transportation",
        max_confidence=0.8,
    ),
    ManualCase(
        "unknown_payee_weird_amount_string",
        {
            "transactionDescription": "??? manual line",
            "transactionAmount": "$1,234.56",
            "account": "salary-checking",
        },
    ),
    ManualCase(
        "bad_amount_string_no_notes",
        {
            "transaction_description": "TARGET STORE",
            "amount": "not-a-number",
            "notes": "",
        },
    ),
    ManualCase(
        "all_whitespace",
        {
            "transaction_description": "   ",
            "merchant_text": "\t",
            "notes": "   ",
            "amount": "",
            "currency": " ",
        },
    ),
    ManualCase(
        "nested_extra_fields",
        {
            "foo": {"bar": 1},
            "metadata": ["x", "y"],
            "amount": "($45.67)",
            "memo": "not mapped",
        },
    ),
    ManualCase(
        "batch_transactions_alias_single_dict",
        {
            "transactions": {
                "transaction_description": "COMCAST CABLE AUTOPAY",
                "amount": -92.0,
            }
        },
        endpoint="/predict_batch",
    ),
]


def response_item(body: dict[str, Any], endpoint: str) -> dict[str, Any]:
    if endpoint == "/predict_batch":
        items = body.get("items")
        if not isinstance(items, list) or len(items) != 1:
            raise AssertionError("predict_batch response must contain one item")
        return items[0]
    return body


def validate_prediction(case: ManualCase, item: dict[str, Any]) -> None:
    top_categories = item.get("top_categories")
    if not isinstance(top_categories, list) or len(top_categories) != 3:
        raise AssertionError(f"{case.name}: expected exactly three top categories")
    if not item.get("predicted_category_id"):
        raise AssertionError(f"{case.name}: missing predicted_category_id")
    if item.get("confidence") != top_categories[0].get("score"):
        raise AssertionError(f"{case.name}: confidence must equal top-1 score")
    if case.expected_top1 and item.get("predicted_category_id") != case.expected_top1:
        raise AssertionError(
            f"{case.name}: expected top1 {case.expected_top1}, "
            f"got {item.get('predicted_category_id')}"
        )
    if case.max_confidence is not None and item.get("confidence", 0) > case.max_confidence:
        raise AssertionError(
            f"{case.name}: expected confidence <= {case.max_confidence}, "
            f"got {item.get('confidence')}"
        )
    for candidate in top_categories:
        if not isinstance(candidate.get("category_id"), str) or not candidate["category_id"]:
            raise AssertionError(f"{case.name}: invalid category id")
        score = candidate.get("score")
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            raise AssertionError(f"{case.name}: invalid score")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run professor-style manual-entry robustness checks against SmartCat serving.",
    )
    parser.add_argument(
        "--base-url",
        default="http://129.114.26.122:30090",
        help="Serving base URL, for example http://host:30090 or https://host:30443/smartcat.",
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--traffic-source",
        default="manual-robustness",
        help="X-Actual-Traffic-Source header value for synthetic/manual tests.",
    )
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification.")
    args = parser.parse_args()

    session = requests.Session()
    headers = {
        "Content-Type": "application/json",
        "X-Actual-Traffic-Source": args.traffic_source,
    }
    base_url = args.base_url.rstrip("/")
    failures = []
    rows = []

    for case in CASES:
        url = f"{base_url}{case.endpoint}"
        try:
            response = session.post(
                url,
                headers=headers,
                json=case.payload,
                timeout=args.timeout,
                verify=not args.insecure,
            )
            body = response.json()
            if response.status_code != 200:
                raise AssertionError(f"HTTP {response.status_code}: {body}")
            item = response_item(body, case.endpoint)
            validate_prediction(case, item)
            rows.append(
                {
                    "case": case.name,
                    "status": response.status_code,
                    "predicted": item["predicted_category_id"],
                    "confidence": item["confidence"],
                    "top3": [candidate["category_id"] for candidate in item["top_categories"]],
                }
            )
        except Exception as exc:
            failures.append({"case": case.name, "error": str(exc)})

    print(json.dumps({"base_url": base_url, "passed": len(rows), "failed": failures, "rows": rows}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
