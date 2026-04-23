# Taxonomy, Calibration, and Registry Update

Date: 2026-04-23

## Scope

This update strengthened three system layers that were previously too loose for
real user interaction and defensible MLOps evaluation:

1. data and label taxonomy alignment
2. confidence calibration plus fallback behavior
3. MLflow registry and promotion/rollback closure

## 1. Data and Label Taxonomy Alignment

### What changed

- Added a canonical deployed taxonomy in `serving/app/taxonomy.py`
- Added safe alias normalization for coarse or inconsistent labels such as:
  - `Bills & Utilities` -> `Utilities & Services`
  - `Healthcare` -> `Healthcare & Medical`
  - `Food` / `Groceries` -> `Food & Dining`
  - `Travel` / `Transit` -> `Transportation`
- Added keyword rules for common merchant and manual-entry patterns
- Added `training/seed_dataset_schema.json` as the canonical seed dataset contract

### Why it matters

This removes ambiguity between:

- generated seed data
- Actual feedback-derived training data
- serving outputs
- dashboard category mixes
- external training repo label spaces

### Important training fix

`training/build_training_set.py` now:

- joins `ml_feedback.final_category_id` back to the `categories` table
- uses the category display name instead of blindly training on a category id
- normalizes labels into the canonical taxonomy
- drops unsupported labels explicitly and records them in the dataset manifest

This is a major correctness fix for feedback-driven retraining.

## 2. Confidence Calibration and Fallback

### What changed

- Added shared confidence post-processing in `serving/app/postprocess.py`
- Added temperature scaling support stored inside model bundle metadata
- Added keyword-based fallback blending for low-confidence or sparse inputs
- Serving now applies confidence policy before constructing Top-3 responses
- Evaluation now uses the same confidence policy, so offline metrics better match online behavior

### Confidence policy

Each trained bundle now includes:

- `confidence_policy.method`
- `confidence_policy.temperature`
- validation calibration metrics:
  - NLL before/after
  - ECE before/after
  - Brier score before/after

It also includes keyword fallback settings for sparse or ambiguous text.

### Why it matters

This makes confidence scores less naive and makes sparse manual-entry behavior
more resilient, especially when the model score is low but the user input still
contains a strong merchant clue like `Verizon`, `Payroll`, `IRS`, or `Starbucks`.

## 3. MLflow Registry and Promotion/Rollback Closure

### What changed

- Training registration now writes:
  - `mlflow_model_name`
  - `mlflow_model_version`
  - `mlflow_run_id`
  - `mlflow_candidate_alias`
  - `register_result.json`
- Registered candidate versions now get:
  - candidate alias
  - model family tag
  - taxonomy version tag
  - confidence temperature tag
- Promotion now:
  - reads `gate_result.json`
  - records gate status in the decision artifact
  - updates the `production` alias
  - tags promoted versions with `actual.role=production`
  - archives prior production metadata when available
- Rollback now:
  - restores archived model metadata
  - updates the `production` alias
  - records replaced and restored metadata in the rollback result

### Why it matters

The system now has a clearer champion/challenger story:

- training creates a registered candidate
- evaluation and gate determine whether it is promotable
- promotion updates both file-based deployment and MLflow registry state
- rollback can restore an archived production model with registry evidence

## Validation

### Unit / contract checks

- `pytest serving/tests/test_feature_adapter.py serving/tests/test_api_contract.py serving/tests/test_taxonomy_postprocess.py`
- `yarn workspace @actual-app/core run test:node src/server/ml/service.test.ts`

### Data builder smoke

A temporary SQLite smoke test verified that:

- feedback category ids are resolved back to category names
- `Bills & Utilities` correctly normalizes to `Utilities & Services`

### Closed-loop pipeline smoke

Ran a full local pipeline using a temporary MLflow SQLite tracking backend:

- dataset quality check: passed
- train: passed
- evaluate: passed
- gate: passed
- export model variants: passed
- register in MLflow: passed
- promote to production alias: passed

Observed outputs:

- taxonomy version persisted into model metadata
- confidence temperature persisted into model metadata
- candidate alias present
- production alias updated successfully
- `register_result.json` written into the deployed bundle

## Bottom Line

This update does not just make the model pipeline look nicer on paper. It
improves the structural integrity of the system:

- the label space is now explicit
- feedback-derived training data is less likely to be wrong
- confidence handling is more realistic
- sparse/manual user inputs have better fallback behavior
- registry-based promotion and rollback are much closer to a real MLOps loop
