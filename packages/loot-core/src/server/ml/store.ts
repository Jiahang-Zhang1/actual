import * as db from '#server/db';
import type { MlPredictResponse } from './types';

let mlTablesInitialized = false;

function makeId(prefix: string) {
  return `${prefix}_${Math.random().toString(36).slice(2)}_${Date.now()}`;
}

async function ensureMlTables() {
  if (mlTablesInitialized) {
    return;
  }

  await db.run(`
    CREATE TABLE IF NOT EXISTS ml_predictions (
      id TEXT PRIMARY KEY,
      transaction_id TEXT NOT NULL,
      model_version TEXT NOT NULL,
      predicted_category_id TEXT NOT NULL,
      confidence REAL NOT NULL,
      top_categories_json TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending',
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `);

  await db.run(`
    CREATE TABLE IF NOT EXISTS ml_feedback (
      id TEXT PRIMARY KEY,
      transaction_id TEXT NOT NULL,
      model_version TEXT NOT NULL,
      predicted_category_id TEXT NOT NULL,
      top_categories_json TEXT NOT NULL,
      final_category_id TEXT NOT NULL,
      feedback_status TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  await db.run(`
    CREATE INDEX IF NOT EXISTS idx_ml_predictions_transaction_id
    ON ml_predictions(transaction_id)
  `);

  await db.run(`
    CREATE INDEX IF NOT EXISTS idx_ml_feedback_transaction_id
    ON ml_feedback(transaction_id)
  `);

  mlTablesInitialized = true;
}

export async function savePrediction(
  transactionId: string,
  prediction: MlPredictResponse,
) {
  await ensureMlTables();

  const now = new Date().toISOString();
  const id = makeId('mlpred');

  await db.run(
    `
      INSERT INTO ml_predictions
        (
          id,
          transaction_id,
          model_version,
          predicted_category_id,
          confidence,
          top_categories_json,
          status,
          created_at,
          updated_at
        )
      VALUES
        (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
    `,
    [
      id,
      transactionId,
      prediction.model_version,
      prediction.predicted_category_id,
      prediction.confidence,
      JSON.stringify(prediction.top_categories),
      now,
      now,
    ],
  );

  console.log('savePrediction called:', {
    transactionId,
    insertedId: id,
    modelVersion: prediction.model_version,
    predictedCategory: prediction.predicted_category_id,
  });

  const verifyRows = await db.all(
    `
      SELECT *
      FROM ml_predictions
      WHERE transaction_id = ?
      ORDER BY created_at DESC
      LIMIT 3
    `,
    [transactionId],
  );

  console.log('savePrediction verifyRows:', transactionId, verifyRows);
}

export async function getLatestPrediction(transactionId: string) {
  await ensureMlTables();

  const row = await db.first(
    `
      SELECT *
      FROM ml_predictions
      WHERE transaction_id = ?
      ORDER BY created_at DESC
      LIMIT 1
    `,
    [transactionId],
  );

  console.log('getLatestPrediction raw row:', transactionId, row);

  return row ?? null;
}

export async function recordFeedback(args: {
  transactionId: string;
  finalCategoryId: string;
}) {
  await ensureMlTables();

  const latest = await getLatestPrediction(args.transactionId);
  if (!latest) {
    return null;
  }

  const topCategories = JSON.parse(latest.top_categories_json || '[]');
  const top1 = latest.predicted_category_id;
  const top3Ids = topCategories.map((x: { category_id: string }) => x.category_id);

  let status = 'overridden';
  if (args.finalCategoryId === top1) {
    status = 'accepted_top1';
  } else if (top3Ids.includes(args.finalCategoryId)) {
    status = 'accepted_top3';
  }

  const now = new Date().toISOString();
  const feedbackId = makeId('mlfb');

  await db.run(
    `
      UPDATE ml_predictions
      SET status = ?, updated_at = ?
      WHERE id = ?
    `,
    [status, now, latest.id],
  );

  await db.run(
    `
      INSERT INTO ml_feedback
        (
          id,
          transaction_id,
          model_version,
          predicted_category_id,
          top_categories_json,
          final_category_id,
          feedback_status,
          created_at
        )
      VALUES
        (?, ?, ?, ?, ?, ?, ?, ?)
    `,
    [
      feedbackId,
      args.transactionId,
      latest.model_version,
      latest.predicted_category_id,
      latest.top_categories_json,
      args.finalCategoryId,
      status,
      now,
    ],
  );

  return { ...latest, feedback_status: status };
}