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
        "shake shack",
        "dominos",
        "coffee",
        "restaurant",
        "dining",
        "subway restaurant",
    ],
    "Transportation": [
        "uber",
        "lyft",
        "nj transit",
        "njt rail",
        "metro north",
        "mta",
        "omny",
        "metro",
        "parking",
        "parkwhiz",
        "park whiz",
        "toll",
        "ezpass",
        "e zpass",
        "train ticket",
        "subway fare",
        "rideshare",
        "taxi",
        "cab",
        "fuel",
        "gas",
        "shell",
        "exxon",
        "chevron",
    ],
    "Shopping & Retail": [
        "amazon",
        "apple store",
        "target",
        "walmart",
        "costco",
        "best buy",
        "ikea",
        "whole foods",
        "trader joe",
        "tj maxx",
        "home depot",
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
        "consolidated edison",
        "coned",
        "xfinity",
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
        "nyc health",
        "citymd",
        "doctor",
        "clinic",
        "medical",
        "health",
    ],
    "Income": [
        "payroll",
        "direct deposit",
        "salary",
        "paycheck",
        "employer payment",
        "wage payment",
        "interest payment",
        "income",
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

ACCOUNT_HINT_RULES = {
    "Charity & Donations": [
        "charity",
        "donation",
        "donor",
        "fundraiser",
        "nonprofit",
    ],
    "Entertainment & Recreation": [
        "entertainment",
        "movie",
        "music",
        "recreation",
        "streaming",
    ],
    "Financial Services": [
        "brokerage",
        "investment",
        "investments",
        "retirement",
        "savings",
    ],
    "Food & Dining": [
        "cafe",
        "coffee",
        "food",
        "grocery",
        "groceries",
        "restaurant",
    ],
    "Government & Legal": [
        "court",
        "government",
        "legal",
        "tax",
    ],
    "Healthcare & Medical": [
        "doctor",
        "health",
        "medical",
        "pharmacy",
    ],
    "Income": [
        "income",
        "payroll",
        "salary",
    ],
    "Shopping & Retail": [
        "retail",
        "shopping",
        "store",
    ],
    "Transportation": [
        "auto",
        "fuel",
        "gas",
        "parking",
        "transit",
        "transport",
        "travel",
    ],
    "Utilities & Services": [
        "cable",
        "electric",
        "internet",
        "mobile",
        "phone",
        "service",
        "utility",
        "utilities",
        "water",
    ],
}

SPARSE_NO_SIGNAL_PRIOR = {
    "Shopping & Retail": 0.34,
    "Food & Dining": 0.27,
    "Utilities & Services": 0.17,
    "Transportation": 0.12,
    "Entertainment & Recreation": 0.10,
}

SPARSE_AMOUNT_PRIORS = {
    "positive:micro": {
        "Financial Services": 0.34,
        "Shopping & Retail": 0.24,
        "Food & Dining": 0.20,
        "Income": 0.14,
        "Utilities & Services": 0.08,
    },
    "positive:small": {
        "Shopping & Retail": 0.26,
        "Transportation": 0.22,
        "Food & Dining": 0.20,
        "Financial Services": 0.18,
        "Income": 0.14,
    },
    "positive:medium": {
        "Shopping & Retail": 0.28,
        "Utilities & Services": 0.22,
        "Financial Services": 0.20,
        "Income": 0.18,
        "Government & Legal": 0.12,
    },
    "positive:large": {
        "Income": 0.60,
        "Financial Services": 0.18,
        "Government & Legal": 0.10,
        "Utilities & Services": 0.08,
        "Shopping & Retail": 0.04,
    },
    "negative:micro": {
        "Food & Dining": 0.30,
        "Entertainment & Recreation": 0.24,
        "Healthcare & Medical": 0.20,
        "Transportation": 0.14,
        "Financial Services": 0.12,
    },
    "negative:small": {
        "Shopping & Retail": 0.28,
        "Transportation": 0.22,
        "Healthcare & Medical": 0.18,
        "Food & Dining": 0.16,
        "Charity & Donations": 0.16,
    },
    "negative:medium": {
        "Utilities & Services": 0.28,
        "Government & Legal": 0.22,
        "Financial Services": 0.18,
        "Shopping & Retail": 0.16,
        "Charity & Donations": 0.16,
    },
    "negative:large": {
        "Government & Legal": 0.34,
        "Financial Services": 0.22,
        "Utilities & Services": 0.18,
        "Shopping & Retail": 0.14,
        "Charity & Donations": 0.12,
    },
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
        "account_hint_rule_categories": sorted(ACCOUNT_HINT_RULES),
        "sparse_no_signal_prior_categories": sorted(SPARSE_NO_SIGNAL_PRIOR),
        "sparse_amount_prior_keys": sorted(SPARSE_AMOUNT_PRIORS),
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


def account_hint_match(account_id: object | None) -> str | None:
    text = normalize_label_key(account_id)
    if not text:
        return None

    best_match: tuple[int, str] | None = None
    for category in CANONICAL_CATEGORIES:
        for keyword in ACCOUNT_HINT_RULES.get(category, []):
            normalized_keyword = normalize_label_key(keyword)
            if normalized_keyword and normalized_keyword in text:
                score = len(normalized_keyword)
                if best_match is None or score > best_match[0]:
                    best_match = (score, category)
    return None if best_match is None else best_match[1]
