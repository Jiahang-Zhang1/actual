from __future__ import annotations

from collections.abc import Iterable
import re


TAXONOMY_VERSION = "2026-04-23"

CANONICAL_CATEGORIES = [
    "Charity & Donations",
    "Entertainment & Recreation",
    "Financial Services",
    "Food & Dining",
    "Government & Legal",
    "Healthcare & Medical",
    "Income",
    "Shopping & Retail",
    "Transportation",
    "Utilities & Services",
]

CANONICAL_CATEGORY_SET = set(CANONICAL_CATEGORIES)

# Keep this mapping deliberately conservative. We only normalize labels when the
# intent is very clear, so we do not quietly collapse unrelated user categories
# into the deployed ML label space.
CATEGORY_ALIASES = {
    "Bills": "Utilities & Services",
    "Bills & Utilities": "Utilities & Services",
    "Dining": "Food & Dining",
    "Entertainment": "Entertainment & Recreation",
    "Food": "Food & Dining",
    "Groceries": "Food & Dining",
    "Healthcare": "Healthcare & Medical",
    "Medical": "Healthcare & Medical",
    "Restaurant": "Food & Dining",
    "Restaurants": "Food & Dining",
    "Salary": "Income",
    "Transit": "Transportation",
    "Travel": "Transportation",
    "Utilities": "Utilities & Services",
}

KEYWORD_RULES = {
    "Food & Dining": [
        "starbucks",
        "chipotle",
        "doordash",
        "grubhub",
        "ubereats",
        "mcdonald",
        "panera",
        "coffee",
        "restaurant",
        "dining",
    ],
    "Transportation": [
        "uber",
        "lyft",
        "nj transit",
        "mta",
        "metro",
        "parking",
        "toll",
        "fuel",
        "gas",
        "shell",
        "exxon",
        "chevron",
    ],
    "Shopping & Retail": [
        "amazon",
        "target",
        "walmart",
        "costco",
        "ikea",
        "marketplace",
        "retail",
        "household",
    ],
    "Entertainment & Recreation": [
        "netflix",
        "spotify",
        "amc",
        "theatre",
        "theater",
        "movie",
        "subscription",
        "recreation",
    ],
    "Utilities & Services": [
        "comcast",
        "con edison",
        "verizon",
        "at&t",
        "att ",
        "electric",
        "utility",
        "cable",
        "water bill",
        "phone bill",
        "internet",
    ],
    "Healthcare & Medical": [
        "cvs",
        "walgreens",
        "pharmacy",
        "quest diagnostics",
        "doctor",
        "clinic",
        "medical",
        "health",
    ],
    "Income": [
        "payroll",
        "direct deposit",
        "salary",
        "interest payment",
        "interest",
        "income",
        "deposit",
    ],
    "Financial Services": [
        "vanguard",
        "bank fee",
        "fee",
        "investment",
        "brokerage",
        "wire transfer",
        "transfer fee",
    ],
    "Charity & Donations": [
        "red cross",
        "donation",
        "charity",
        "nonprofit",
        "fundraiser",
    ],
    "Government & Legal": [
        "irs",
        "tax",
        "government",
        "dmv",
        "legal",
        "court",
    ],
}


def normalize_label_key(value: object | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().casefold()).strip()


def canonicalize_category(value: object | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text in CANONICAL_CATEGORY_SET:
        return text

    alias = CATEGORY_ALIASES.get(text)
    if alias:
        return alias

    normalized_key = normalize_label_key(text)
    for candidate in CANONICAL_CATEGORIES:
        if normalize_label_key(candidate) == normalized_key:
            return candidate
    for alias_name, canonical in CATEGORY_ALIASES.items():
        if normalize_label_key(alias_name) == normalized_key:
            return canonical
    return None


def canonicalize_categories(values: Iterable[object | None]) -> list[str | None]:
    return [canonicalize_category(value) for value in values]


def taxonomy_manifest() -> dict:
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "canonical_categories": CANONICAL_CATEGORIES,
        "aliases": CATEGORY_ALIASES,
        "keyword_rule_categories": sorted(KEYWORD_RULES),
    }


def keyword_rule_match(description: object | None) -> str | None:
    text = normalize_label_key(description)
    if not text:
        return None

    best_match: tuple[int, str] | None = None
    for category in CANONICAL_CATEGORIES:
        for keyword in KEYWORD_RULES.get(category, []):
            normalized_keyword = normalize_label_key(keyword)
            if normalized_keyword and normalized_keyword in text:
                score = len(normalized_keyword)
                if best_match is None or score > best_match[0]:
                    best_match = (score, category)
    return None if best_match is None else best_match[1]
