import numpy as np

from app.postprocess import apply_confidence_policy, temperature_scale_probabilities
from app.taxonomy import account_hint_match, canonicalize_category, keyword_rule_match


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


def test_account_hint_match_finds_category_from_account_name():
    assert account_hint_match("utilities-savings-wallet") == "Utilities & Services"
    assert account_hint_match("salary-checking") == "Income"


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


def test_confidence_policy_uses_amount_based_sparse_prior_for_manual_entry():
    classes = [
        "Income",
        "Financial Services",
        "Government & Legal",
        "Shopping & Retail",
    ]
    raw = np.array([0.10, 0.20, 0.25, 0.45], dtype=float)
    adjusted = apply_confidence_policy(
        raw,
        classes,
        description="manual entry",
        description_source="derived",
        amount=2450.0,
        account_id="checking-account",
        metadata={"confidence_policy": {}},
    )
    assert classes[int(np.argmax(adjusted))] == "Income"


def test_confidence_policy_overrides_no_signal_healthcare_bias_with_conservative_prior():
    classes = [
        "Healthcare & Medical",
        "Shopping & Retail",
        "Food & Dining",
        "Utilities & Services",
        "Transportation",
    ]
    raw = np.array([0.62, 0.10, 0.09, 0.10, 0.09], dtype=float)
    adjusted = apply_confidence_policy(
        raw,
        classes,
        description="manual entry",
        description_source="derived",
        metadata={"confidence_policy": {}},
    )
    ordered = [classes[int(index)] for index in np.argsort(adjusted)[::-1][:3]]
    assert ordered[:3] == [
        "Shopping & Retail",
        "Food & Dining",
        "Utilities & Services",
    ]
    assert float(np.max(adjusted)) < 0.4
