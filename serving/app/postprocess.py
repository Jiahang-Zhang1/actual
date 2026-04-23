from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from app.taxonomy import keyword_rule_match
except ModuleNotFoundError:  # pragma: no cover - script entrypoints import via serving.app
    from serving.app.taxonomy import keyword_rule_match

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
    blend_weight = float(keyword_policy.get("blend_weight", 0.35))
    blend_weight = min(max(blend_weight, 0.0), 1.0)

    if top1_confidence > max_primary_confidence and int(np.argmax(matrix)) == target_idx:
        return matrix
    if top1_confidence > max_primary_confidence and description_source not in {"notes", "derived"}:
        return matrix

    boost = np.zeros_like(matrix)
    boost[target_idx] = 1.0
    blended = ((1.0 - blend_weight) * matrix) + (blend_weight * boost)
    return _normalize_rows(blended)


def apply_confidence_policy(
    probabilities: np.ndarray,
    classes: list[str],
    *,
    description: str,
    description_source: str,
    metadata: Mapping[str, Any] | None,
) -> np.ndarray:
    policy = dict((metadata or {}).get("confidence_policy") or {})
    matrix = np.asarray(probabilities, dtype=float).reshape(1, -1)
    calibrated = temperature_scale_probabilities(matrix, policy.get("temperature"))
    blended = blend_keyword_rule(
        calibrated[0],
        classes,
        description,
        description_source,
        policy.get("keyword_fallback"),
    )
    return np.asarray(blended, dtype=float).reshape(-1)
