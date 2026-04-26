import { logger } from '#platform/server/log';
import * as db from '#server/db';

import { sendFeedback } from './service';
import type { MlPredictionStatus, MlPredictResponse } from './types';

function parseTopCategories(topCategoriesJson: null | string | undefined) {
  try {
    const parsed = JSON.parse(topCategoriesJson || '[]');
    return Array.isArray(parsed)
      ? parsed.filter(
          item =>
            item &&
            typeof item === 'object' &&
            'category_id' in item &&
            'score' in item,
        )
      : [];
  } catch {
    return [];
  }
}

function normalizeCategoryKey(value: null | string | undefined) {
  return (value ?? '')
    .replace(/&/g, 'and')
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .trim()
    .toLowerCase();
}

export async function savePrediction(
  transactionId: string,
  prediction: MlPredictResponse,
  status: MlPredictionStatus = 'pending',
) {
  const now = new Date().toISOString();

  db.runQuery(
    `
      INSERT INTO ml_predictions (
        transaction_id,
        model_version,
        predicted_category_id,
        confidence,
        top_categories_json,
        status,
        created_at,
        updated_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `,
    [
      transactionId,
      prediction.model_version,
      prediction.predicted_category_id,
      prediction.confidence,
      JSON.stringify(prediction.top_categories),
      status,
      now,
      now,
    ],
  );
}

export async function getLatestPrediction(transactionId: string) {
  const rows = db.runQuery(
    `
      SELECT *
      FROM ml_predictions
      WHERE transaction_id = ?
      ORDER BY created_at DESC
      LIMIT 1
    `,
    [transactionId],
  );

  return rows?.[0] ?? null;
}

export async function recordFeedback(args: {
  transactionId: string;
  finalCategoryId: string;
  prediction?: MlPredictResponse | null;
  syncToServing?: boolean;
}) {
  let latest = await getLatestPrediction(args.transactionId);
  if (!latest && args.prediction) {
    await savePrediction(args.transactionId, args.prediction);
    latest = await getLatestPrediction(args.transactionId);
  }

  if (!latest) {
    return null;
  }

  const topCategories = parseTopCategories(latest.top_categories_json);
  const top1CategoryId = latest.predicted_category_id;
  const top3CategoryIds = topCategories.map(
    (item: { category_id: string }) => item.category_id,
  );
  const finalCategory = await db.getCategory(args.finalCategoryId);
  const finalCategoryKeys = new Set([
    normalizeCategoryKey(args.finalCategoryId),
    normalizeCategoryKey(finalCategory?.name),
  ]);
  const matchesFinalCategory = (categoryIdOrName: string) =>
    finalCategoryKeys.has(normalizeCategoryKey(categoryIdOrName));

  let status: MlPredictionStatus = 'overridden';
  if (matchesFinalCategory(top1CategoryId)) {
    status = 'accepted_top1';
  } else if (top3CategoryIds.some(matchesFinalCategory)) {
    status = 'accepted_top3';
  }

  const now = new Date().toISOString();

  db.runQuery(
    `
      UPDATE ml_predictions
      SET status = ?, updated_at = ?
      WHERE id = ?
    `,
    [status, now, latest.id],
  );

  db.runQuery(
    `
      INSERT INTO ml_feedback (
        transaction_id,
        model_version,
        predicted_category_id,
        top_categories_json,
        final_category_id,
        feedback_status,
        created_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `,
    [
      args.transactionId,
      latest.model_version,
      latest.predicted_category_id,
      latest.top_categories_json,
      args.finalCategoryId,
      status,
      now,
    ],
  );

  const result = {
    ...latest,
    feedback_status: status,
  };

  if (args.syncToServing !== false) {
    try {
      // Mirror feedback to serving so monitoring/alerts are based on live user
      // behavior, while still keeping local SQLite as the source of truth.
      await sendFeedback({
        transactionId: args.transactionId,
        modelVersion: latest.model_version,
        predictedCategoryId: latest.predicted_category_id,
        // Serving monitors label-level acceptance, while Actual stores category ids.
        appliedCategoryId: finalCategory?.name ?? args.finalCategoryId,
        confidence: latest.confidence,
        candidateCategoryIds: top3CategoryIds,
      });
    } catch (error) {
      logger.error('ML feedback sync to serving failed', error);
    }
  }

  return result;
}
