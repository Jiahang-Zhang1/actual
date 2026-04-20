CREATE TABLE IF NOT EXISTS ml_predictions (
  id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
  transaction_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  predicted_category_id TEXT NOT NULL,
  confidence REAL NOT NULL,
  top_categories_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ml_predictions_transaction_id_created_at_idx
  ON ml_predictions (transaction_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ml_predictions_status_idx
  ON ml_predictions (status);

CREATE TABLE IF NOT EXISTS ml_feedback (
  id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
  transaction_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  predicted_category_id TEXT NOT NULL,
  top_categories_json TEXT NOT NULL,
  final_category_id TEXT NOT NULL,
  feedback_status TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ml_feedback_transaction_id_created_at_idx
  ON ml_feedback (transaction_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ml_feedback_status_idx
  ON ml_feedback (feedback_status);
