import numpy as np

from app.postprocess import apply_confidence_policy, temperature_scale_probabilities
from app.taxonomy import canonicalize_category, keyword_rule_match


def test_canonicalize_category_maps_safe_aliases():
    assert canonicalize_category("Bills & Utilities") == "Utilities & Services"
    assert canonicalize_category("Healthcare") == "Healthcare & Medical"
    assert canonicalize_category("Food") == "Food & Dining"
    assert canonicalize_category("Unknown User Label") is None


def test_temperature_scaling_reduces_overconfident_top1_score():
    probabilities = np.array([[0.90, 0.05, 0.05]], dtype=float)
    scaled = temperature_scale_probabilities(probabilities, 2.0)
    assert scaled.shape == probabilities.shape
    assert float(scaled[0][0]) < 0.90
    assert np.isclose(float(np.sum(scaled[0])), 1.0)


def test_keyword_rule_match_finds_canonical_category():
    assert keyword_rule_match("Monthly Verizon phone bill autopay") == "Utilities & Services"
    assert keyword_rule_match("Payroll direct deposit from employer") == "Income"


def test_confidence_policy_boosts_rule_match_for_sparse_notes():
    classes = [
        "Utilities & Services",
        "Food & Dining",
        "Shopping & Retail",
    ]
    raw = np.array([0.34, 0.33, 0.33], dtype=float)
    adjusted = apply_confidence_policy(
        raw,
        classes,
        description="monthly verizon phone bill",
        description_source="notes",
        metadata={
            "confidence_policy": {
                "temperature": 1.4,
                "keyword_fallback": {
                    "enabled": True,
                    "blend_weight": 0.35,
                    "max_primary_confidence": 0.58,
                    "allowed_sources": ["notes", "derived"],
                },
            }
        },
    )
    assert classes[int(np.argmax(adjusted))] == "Utilities & Services"
    assert float(adjusted[0]) > float(raw[0])
