import * as db from '#server/db';
import type { MlPredictResponse } from './types';

export async function savePrediction(
  transactionId: string,
  prediction: MlPredictResponse,
) {
  const now = new Date().toISOString();

  await db.runQuery(`
    INSERT INTO ml_predictions
      (transaction_id, model_version, predicted_category_id, confidence, top_categories_json, status, created_at, updated_at)
    VALUES
      (?, ?, ?, ?, ?, 'pending', ?, ?)
  `, [
    transactionId,
    prediction.model_version,
    prediction.predicted_category_id,
    prediction.confidence,
    JSON.stringify(prediction.top_categories),
    now,
    now,
  ]);
}

export async function getLatestPrediction(transactionId: string) {
  const rows = await db.runQuery(`
    SELECT *
    FROM ml_predictions
    WHERE transaction_id = ?
    ORDER BY created_at DESC
    LIMIT 1
  `, [transactionId]);

  return rows?.[0] ?? null;
}

export async function recordFeedback(args: {
  transactionId: string;
  finalCategoryId: string;
}) {
  const latest = await getLatestPrediction(args.transactionId);
  if (!latest) {
    return null;
  }

  const topCategories = JSON.parse(latest.top_categories_json || '[]');
  const top1 = latest.predicted_category_id;
  const top3Ids = topCategories.map(x => x.category_id);

  let status = 'overridden';
  if (args.finalCategoryId === top1) {
    status = 'accepted_top1';
  } else if (top3Ids.includes(args.finalCategoryId)) {
    status = 'accepted_top3';
  }

  const now = new Date().toISOString();

  await db.runQuery(`
    UPDATE ml_predictions
    SET status = ?, updated_at = ?
    WHERE id = ?
  `, [status, now, latest.id]);

  await db.runQuery(`
    INSERT INTO ml_feedback
      (transaction_id, model_version, predicted_category_id, top_categories_json, final_category_id, feedback_status, created_at)
    VALUES
      (?, ?, ?, ?, ?, ?, ?)
  `, [
    args.transactionId,
    latest.model_version,
    latest.predicted_category_id,
    latest.top_categories_json,
    args.finalCategoryId,
    status,
    now,
  ]);

  return { ...latest, feedback_status: status };
}
