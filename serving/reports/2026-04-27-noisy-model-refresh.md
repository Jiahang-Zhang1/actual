# 2026-04-27 Noisy Model Refresh

## Why This Update Was Needed

The demo surfaced a realistic manual-entry failure mode: `lyft 18.2 ride home` should be Transportation, not Income. The deployed API fallback already corrected the example, but the old model was trained on a very clean 225-row dataset with perfect metrics. That made the system brittle for professor-style manual testing.

## What Changed

- Rebuilt the synthetic data generator so it creates import files, API batch payloads, and training CSVs from the same noisy taxonomy-aligned source.
- Added data-team merchant coverage including `LYFT`, `UBER`, `METRO NORTH`, `MTA`, `PARKWHIZ`, `TARGET`, `COSTCO`, `BEST BUY`, `VERIZON`, `CONSOLIDATED EDISON`, `SPOTIFY`, `CVS`, `WALGREENS`, and `NYC HEALTH`.
- Added realistic noise: blank notes, blank amounts, odd casing, truncated payees, account-only fallback rows, sign conflicts, and payee/notes conflicts.
- Expanded keyword and sparse fallback rules for transportation, utilities, retail, food, and healthcare.
- Added regression cases for Lyft, Uber, and MTA manual-entry payloads.
- Updated promotion tooling with an explicit `threshold-only` comparison mode for intentional dataset refreshes where old champion metrics are not comparable.
- Registered the refreshed model in Chameleon MLflow as `actual-smart-transaction-categorizer` version `1` and set the `production` alias.

## Final Model

- Model version: `v20260427184544`
- Model family: `logreg`
- Validation rows: `360`
- Validation top1/top3/macro-F1: `0.9667 / 0.9833 / 0.9687`
- Test rows: `360`
- Test top1/top3/macro-F1: `0.9861 / 0.9944 / 0.9869`
- High-confidence precision: `1.0`
- ONNX dynamic quantized label parity: `1.0`
- ONNX dynamic quantized Top-3 parity: `0.9766`

## Key Manual Regression

Payload:

```json
{
  "transaction_description": "lyft 18.2 ride home",
  "amount": -18.2,
  "notes": "",
  "currency": "USD",
  "country": "US"
}
```

Local deployed-bundle result:

```text
Transportation, confidence 0.794511
Top-3: Transportation, Food & Dining, Entertainment & Recreation
```

## Commands Used

```bash
python3 scripts/generate_actual_bank_import.py \
  --rows 2400 \
  --seed 9183 \
  --training-output artifacts/test-data/noisy_training_transactions.csv \
  --csv-output artifacts/test-data/noisy_actual_bank_transactions.csv \
  --qif-output artifacts/test-data/noisy_actual_bank_transactions.qif \
  --batch-output serving/runtime/noisy_batch_input_realistic.json

MLFLOW_TRACKING_URI=http://129.114.26.122:8000 \
MLFLOW_REGISTER_MODEL_NAME=actual-smart-transaction-categorizer \
python3 scripts/run_mlops_pipeline.py \
  --workspace-dir artifacts/mlops-pipeline/noisy-final \
  --source-training-csv artifacts/test-data/noisy_training_transactions.csv \
  --deployed-dir serving/runtime/deployed \
  --archive-dir serving/runtime/archive \
  --min-top1-accuracy 0.55 \
  --min-top3-accuracy 0.75 \
  --min-macro-f1 0.45 \
  --min-high-confidence-precision 0.65 \
  --min-mapped-top1-accuracy 0.55 \
  --per-class-recall-min 0.20 \
  --per-class-support-min 5 \
  --high-confidence-threshold 0.65 \
  --model-families logreg \
  --promotion-comparison-mode threshold-only
```

## Verification

```bash
PYTHONPATH=serving python3 -m pytest serving/tests data/tests -q
```

Result: `34 passed`.
