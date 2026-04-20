# Smart Transaction Categorization Safeguarding Plan

This document records the mechanisms implemented for the system implementation
milestone.

## Fairness

Mechanism: monitor prediction and feedback quality by data segment where
available. The data quality job emits distribution checks for `country` and
`currency`, while serving records selected category counts and top-1/top-3
acceptance. These metrics are visible through Prometheus via serving `/metrics`.

Evidence:

- `data/data_quality_check.py`
- `serving/app/main.py`
- `k8s/ml-system/base/data-quality-cronjob.yaml`

## Privacy

Mechanism: serving logs do not persist raw transaction descriptions. Runtime
prediction logs store category, confidence, candidate category ids, model
version, and timestamps. Feedback logs store transaction ids and category ids
needed for accountability and retraining.

Evidence:

- `serving/runtime/prediction_events.jsonl`
- `serving/runtime/feedback_events.jsonl`
- `packages/loot-core/src/server/ml/store.ts`

## Robustness

Mechanism: the frontend uses batch prediction first and single prediction as a
fallback. Serving exposes health and readiness checks. Rollback is automated
when production thresholds fail.

Evidence:

- `packages/desktop-client/src/components/accounts/Account.tsx`
- `packages/loot-core/src/server/ml/app.ts`
- `serving/tools/execute_rollout_action.py`
- `k8s/ml-system/base/rollout-decision-cronjob.yaml`

## Explainability And Transparency

Mechanism: users see the top-3 candidate categories with confidence scores in
the regular transaction flow. The selected AI suggestion is highlighted, and
users can override it with one click.

Evidence:

- `packages/desktop-client/src/components/transactions/TransactionsTable.tsx`
- `serving/app/main.py`

## Accountability

Mechanism: all user overrides are written to `ml_feedback`, mirrored to serving,
and included in monitoring summaries. Promotion and rollback decisions are
written as JSON evidence in the model archive.

Evidence:

- `packages/loot-core/src/server/ml/store.ts`
- `scripts/promote_model.py`
- `scripts/rollback_model.py`
- `scripts/simulate_promotion_rollback.py`

## Human Control

Mechanism: automatic Top-1 categorization only applies to uncategorized rows.
Users can choose any Top-3 alternative, and that selection replaces the
category and becomes retraining feedback. Canary promotion can be automated,
but the same scripts support dry-run review before production execution.
