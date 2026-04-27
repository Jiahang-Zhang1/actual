from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from app.taxonomy import (
        SPARSE_AMOUNT_PRIORS,
        SPARSE_NO_SIGNAL_PRIOR,
        account_hint_match,
        keyword_rule_match,
    )
except ModuleNotFoundError:  # pragma: no cover - script entrypoints import via serving.app
    from serving.app.taxonomy import (
        SPARSE_AMOUNT_PRIORS,
        SPARSE_NO_SIGNAL_PRIOR,
        account_hint_match,
        keyword_rule_match,
    )

EPSILON = 1e-9


def _softmax_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix - np.max(matrix, axis=1, keepdims=True)
    exp_matrix = np.exp(matrix)
    denom = np.sum(exp_matrix, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return exp_matrix / denom


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    was_1d = matrix.ndim == 1
    if was_1d:
        matrix = matrix.reshape(1, -1)
    matrix = np.clip(matrix, EPSILON, None)
    normalized = matrix / np.sum(matrix, axis=1, keepdims=True)
    return normalized[0] if was_1d else normalized


def _bundle_metadata_path(model_bundle_dir: str) -> Path:
    return Path(model_bundle_dir) / "metadata.json"


@lru_cache(maxsize=4)
def _load_bundle_metadata_cached(path_str: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle_metadata(model_bundle_dir: str) -> dict[str, Any]:
    path = _bundle_metadata_path(model_bundle_dir)
    if not path.exists():
        return {}
    return _load_bundle_metadata_cached(str(path.resolve()), path.stat().st_mtime_ns)


def temperature_scale_probabilities(probabilities: np.ndarray, temperature: float | None) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if temperature is None or temperature <= 0 or np.isclose(temperature, 1.0):
        return _normalize_rows(matrix)

    log_probs = np.log(np.clip(matrix, EPSILON, 1.0))
    return _softmax_rows(log_probs / float(temperature))


def blend_keyword_rule(
    probabilities: np.ndarray,
    classes: list[str],
    description: str,
    description_source: str,
    keyword_policy: Mapping[str, Any] | None,
) -> np.ndarray:
    if not keyword_policy or not keyword_policy.get("enabled", False):
        return probabilities

    matched_category = keyword_rule_match(description)
    if not matched_category or matched_category not in classes:
        return probabilities

    allowed_sources = set(keyword_policy.get("allowed_sources") or [])
    if allowed_sources and description_source not in allowed_sources:
        return probabilities

    matrix = np.asarray(probabilities, dtype=float).copy()
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    target_idx = class_to_index[matched_category]
    top1_confidence = float(np.max(matrix))
    max_primary_confidence = float(keyword_policy.get("max_primary_confidence", 0.55))
    high_confidence_override = float(keyword_policy.get("high_confidence_override", 0.82))
    blend_weight = float(keyword_policy.get("blend_weight", 0.35))
    blend_weight = min(max(blend_weight, 0.0), 1.0)

    if top1_confidence > max_primary_confidence and int(np.argmax(matrix)) == target_idx:
        return matrix
    if (
        top1_confidence > high_confidence_override
        and description_source not in {"notes", "derived"}
    ):
        return matrix

    boost = np.zeros_like(matrix)
    boost[target_idx] = 1.0
    blended = ((1.0 - blend_weight) * matrix) + (blend_weight * boost)
    return _normalize_rows(blended)


def _amount_prior_key(amount: float | None) -> str | None:
    if amount is None:
        return None
    try:
        numeric = float(amount)
    except (TypeError, ValueError):
        return None
    if np.isclose(numeric, 0.0):
        return None

    magnitude = abs(numeric)
    if magnitude < 20:
        bucket = "micro"
    elif magnitude < 100:
        bucket = "small"
    elif magnitude < 500:
        bucket = "medium"
    else:
        bucket = "large"
    sign = "positive" if numeric > 0 else "negative"
    return f"{sign}:{bucket}"


def _distribution_for_classes(
    classes: list[str],
    distribution: Mapping[str, float] | None,
) -> np.ndarray | None:
    if not distribution:
        return None
    vector = np.zeros(len(classes), dtype=float)
    for index, label in enumerate(classes):
        vector[index] = float(distribution.get(label, 0.0))
    if np.allclose(vector.sum(), 0.0):
        return None
    return _normalize_rows(vector)


def _amount_prior_vector(
    classes: list[str],
    amount: float | None,
    policy: Mapping[str, Any] | None,
) -> np.ndarray | None:
    amount_key = _amount_prior_key(amount)
    if not policy or not amount_key:
        return None
    return _distribution_for_classes(
        classes,
        (policy.get("amount_priors") or {}).get(amount_key),
    )


def blend_amount_conflict_prior(
    probabilities: np.ndarray,
    classes: list[str],
    *,
    description: str,
    description_source: str,
    amount: float | None,
    amount_conflict_policy: Mapping[str, Any] | None,
) -> np.ndarray:
    if not amount_conflict_policy or not amount_conflict_policy.get("enabled", False):
        return probabilities

    allowed_sources = set(amount_conflict_policy.get("allowed_sources") or [])
    if allowed_sources and description_source not in allowed_sources:
        return probabilities

    keyword_category = keyword_rule_match(description)
    if not keyword_category or keyword_category not in classes:
        return probabilities

    amount_prior = _amount_prior_vector(classes, amount, amount_conflict_policy)
    if amount_prior is None:
        return probabilities

    amount_category = classes[int(np.argmax(amount_prior))]
    if amount_category == keyword_category:
        return probabilities

    blend_weight = float(amount_conflict_policy.get("blend_weight", 0.18))
    blend_weight = min(max(blend_weight, 0.0), 1.0)
    matrix = np.asarray(probabilities, dtype=float).copy()
    blended = ((1.0 - blend_weight) * matrix) + (blend_weight * amount_prior)
    return _normalize_rows(blended)


def blend_sparse_prior(
    probabilities: np.ndarray,
    classes: list[str],
    *,
    description_source: str,
    amount: float | None,
    account_id: str | None,
    sparse_policy: Mapping[str, Any] | None,
) -> np.ndarray:
    if not sparse_policy or not sparse_policy.get("enabled", False):
        return probabilities

    allowed_sources = set(sparse_policy.get("allowed_sources") or [])
    if allowed_sources and description_source not in allowed_sources:
        return probabilities

    matrix = np.asarray(probabilities, dtype=float).copy()
    class_to_index = {label: idx for idx, label in enumerate(classes)}

    amount_prior_vector = _amount_prior_vector(classes, amount, sparse_policy)
    account_category = account_hint_match(account_id)
    account_prior_vector = None
    if account_category and account_category in class_to_index:
        account_prior_vector = np.zeros(len(classes), dtype=float)
        account_prior_vector[class_to_index[account_category]] = 1.0
        account_prior_vector = _normalize_rows(account_prior_vector)
    default_prior_vector = _distribution_for_classes(classes, sparse_policy.get("default_prior"))
    if default_prior_vector is None:
        return matrix

    prior = np.zeros(len(classes), dtype=float)
    if amount_prior_vector is not None:
        prior += float(sparse_policy.get("amount_prior_weight", 0.7)) * amount_prior_vector
    if account_prior_vector is not None:
        prior += float(sparse_policy.get("account_prior_weight", 0.25)) * account_prior_vector
    if np.allclose(prior.sum(), 0.0):
        prior = default_prior_vector.copy()
    else:
        prior += float(sparse_policy.get("default_prior_weight", 0.2)) * default_prior_vector
        prior = _normalize_rows(prior)

    no_signal = amount_prior_vector is None and account_prior_vector is None
    blend_weight = float(
        sparse_policy.get("no_signal_blend_weight" if no_signal else "blend_weight", 0.6)
    )
    blend_weight = min(max(blend_weight, 0.0), 1.0)
    max_primary_confidence = float(sparse_policy.get("max_primary_confidence", 0.62))

    if (
        float(np.max(matrix)) > max_primary_confidence
        and amount_prior_vector is None
        and account_prior_vector is None
    ):
        blend_weight = max(blend_weight, float(sparse_policy.get("high_confidence_no_signal_blend_weight", 0.92)))

    blended = ((1.0 - blend_weight) * matrix) + (blend_weight * prior)
    return _normalize_rows(blended)


def default_confidence_policy() -> dict[str, Any]:
    return {
        "method": "temperature_keyword_amount_conflict_sparse_fallback",
        "temperature": 1.0,
        "keyword_fallback": {
            "enabled": True,
            "blend_weight": 0.55,
            "max_primary_confidence": 0.58,
            "high_confidence_override": 0.82,
            "allowed_sources": [
                "transaction_description",
                "transaction_description_clean",
                "merchant_text",
                "imported_description",
                "notes",
                "derived",
            ],
        },
        "amount_conflict_fallback": {
            "enabled": True,
            "blend_weight": 0.18,
            "allowed_sources": [
                "transaction_description",
                "transaction_description_clean",
                "merchant_text",
                "imported_description",
                "notes",
            ],
            "amount_priors": SPARSE_AMOUNT_PRIORS,
        },
        "sparse_fallback": {
            "enabled": True,
            "allowed_sources": ["derived"],
            "blend_weight": 0.68,
            "no_signal_blend_weight": 0.94,
            "high_confidence_no_signal_blend_weight": 0.97,
            "max_primary_confidence": 0.62,
            "amount_prior_weight": 0.72,
            "account_prior_weight": 0.25,
            "default_prior_weight": 0.22,
            "default_prior": SPARSE_NO_SIGNAL_PRIOR,
            "amount_priors": SPARSE_AMOUNT_PRIORS,
        },
    }


def apply_confidence_policy(
    probabilities: np.ndarray,
    classes: list[str],
    *,
    description: str,
    description_source: str,
    amount: float | None = None,
    account_id: str | None = None,
    metadata: Mapping[str, Any] | None,
) -> np.ndarray:
    policy = default_confidence_policy()
    policy.update(dict((metadata or {}).get("confidence_policy") or {}))
    matrix = np.asarray(probabilities, dtype=float).reshape(1, -1)
    calibrated = temperature_scale_probabilities(matrix, policy.get("temperature"))
    blended = blend_keyword_rule(
        calibrated[0],
        classes,
        description,
        description_source,
        policy.get("keyword_fallback"),
    )
    blended = blend_amount_conflict_prior(
        blended,
        classes,
        description=description,
        description_source=description_source,
        amount=amount,
        amount_conflict_policy=policy.get("amount_conflict_fallback"),
    )
    blended = blend_sparse_prior(
        blended,
        classes,
        description_source=description_source,
        amount=amount,
        account_id=account_id,
        sparse_policy=policy.get("sparse_fallback"),
    )
    return np.asarray(blended, dtype=float).reshape(-1)
