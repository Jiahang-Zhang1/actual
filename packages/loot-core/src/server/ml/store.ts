import { logger } from '#platform/server/log';
import * as db from '#server/db';

import { sendFeedback } from './service';
import type {
  MlPredictionRecord,
  MlPredictionStatus,
  MlPredictResponse,
} from './types';

let ensuredDatabase: unknown = null;

function ensureMlTables() {
  const database = db.getDatabase();
  if (ensuredDatabase === database) {
    return;
  }

  db.execQuery(`
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
  `);

  ensuredDatabase = database;
}

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
  const normalized = (value ?? '')
    .replace(/&/g, 'and')
    .replace(/[^a-zA-Z0-9]+/g, ' ')
    .trim()
    .toLowerCase();

  const aliases: Record<string, string> = {
    'bank fees': 'financial services',
    charity: 'charity and donations',
    cell: 'utilities and services',
    clothing: 'shopping and retail',
    doctor: 'healthcare and medical',
    electric: 'utilities and services',
    electricity: 'utilities and services',
    entertainment: 'entertainment and recreation',
    fees: 'financial services',
    financial: 'financial services',
    food: 'food and dining',
    gift: 'charity and donations',
    'gov legal': 'government and legal',
    healthcare: 'healthcare and medical',
    internet: 'utilities and services',
    medical: 'healthcare and medical',
    pharmacy: 'healthcare and medical',
    power: 'utilities and services',
    restaurants: 'food and dining',
    shopping: 'shopping and retail',
    taxes: 'government and legal',
    transport: 'transportation',
    transit: 'transportation',
    utilities: 'utilities and services',
    water: 'utilities and services',
  };

  return aliases[normalized] ?? normalized;
}

export async function savePrediction(
  transactionId: string,
  prediction: MlPredictResponse,
  status: MlPredictionStatus = 'pending',
) {
  ensureMlTables();

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
  ensureMlTables();

  const rows = db.runQuery<MlPredictionRecord>(
    `
      SELECT *
      FROM ml_predictions
      WHERE transaction_id = ?
      ORDER BY created_at DESC
      LIMIT 1
    `,
    [transactionId],
    true,
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
  const appliedServingCategoryId =
    topCategories.find((item: { category_id: string }) =>
      matchesFinalCategory(item.category_id),
    )?.category_id ??
    finalCategory?.name ??
    args.finalCategoryId;

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
        appliedCategoryId: appliedServingCategoryId,
        confidence: latest.confidence,
        candidateCategoryIds: top3CategoryIds,
      });
    } catch (error) {
      logger.error('ML feedback sync to serving failed', error);
    }
  }

  return result;
}
