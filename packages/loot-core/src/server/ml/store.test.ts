import { beforeEach, describe, expect, it, vi } from 'vitest';

import { sendFeedback } from './service';
import { getLatestPrediction, recordFeedback } from './store';
import type { MlPredictionRecord, MlPredictResponse } from './types';

const mocks = vi.hoisted(() => ({
  database: {},
  execQuery: vi.fn(),
  getCategory: vi.fn(),
  getDatabase: vi.fn(),
  runQuery: vi.fn(),
  sendFeedback: vi.fn(),
}));

vi.mock('#server/db', () => ({
  execQuery: mocks.execQuery,
  getCategory: mocks.getCategory,
  getDatabase: mocks.getDatabase,
  runQuery: mocks.runQuery,
}));

vi.mock('./service', () => ({
  sendFeedback: mocks.sendFeedback,
}));

describe('ml prediction store', () => {
  beforeEach(() => {
    mocks.database = {};
    mocks.execQuery.mockReset();
    mocks.getCategory.mockReset();
    mocks.getDatabase.mockReset();
    mocks.runQuery.mockReset();
    mocks.sendFeedback.mockReset();

    mocks.getDatabase.mockImplementation(() => mocks.database);
    mocks.getCategory.mockResolvedValue(null);
    mocks.sendFeedback.mockResolvedValue({ saved: true, status: 'ok' });
  });

  it('reads the latest prediction rows instead of a mutation result', async () => {
    const latest: MlPredictionRecord = {
      id: 'prediction-id',
      transaction_id: 'transaction-id',
      model_version: 'v-test',
      predicted_category_id: 'Food & Dining',
      confidence: 0.81,
      top_categories_json: JSON.stringify([
        { category_id: 'Food & Dining', score: 0.81 },
      ]),
      status: 'pending',
      created_at: '2026-04-27T00:00:00.000Z',
      updated_at: '2026-04-27T00:00:00.000Z',
    };
    mocks.runQuery.mockReturnValueOnce([latest]);

    await expect(getLatestPrediction('transaction-id')).resolves.toEqual(
      latest,
    );
    expect(mocks.runQuery.mock.calls[0][2]).toBe(true);
  });

  it('syncs feedback using model category labels when local aliases match', async () => {
    let latest: MlPredictionRecord | null = null;
    const prediction: MlPredictResponse = {
      predicted_category_id: 'Utilities & Services',
      confidence: 0.7,
      top_categories: [
        { category_id: 'Utilities & Services', score: 0.7 },
        { category_id: 'Government & Legal', score: 0.08 },
        { category_id: 'Financial Services', score: 0.05 },
      ],
      model_version: 'v-test',
    };

    mocks.getCategory.mockResolvedValue({ id: 'power-id', name: 'Power' });
    mocks.runQuery.mockImplementation((sql, params, fetchAll) => {
      const query = String(sql);
      if (query.includes('SELECT *')) {
        return fetchAll && latest ? [latest] : [];
      }

      if (query.includes('INSERT INTO ml_predictions')) {
        latest = {
          id: 'prediction-id',
          transaction_id: String(params?.[0]),
          model_version: String(params?.[1]),
          predicted_category_id: String(params?.[2]),
          confidence: Number(params?.[3]),
          top_categories_json: String(params?.[4]),
          status: String(params?.[5]) as MlPredictionRecord['status'],
          created_at: String(params?.[6]),
          updated_at: String(params?.[7]),
        };
      }

      if (query.includes('UPDATE ml_predictions') && latest) {
        latest = {
          ...latest,
          status: String(params?.[0]) as MlPredictionRecord['status'],
          updated_at: String(params?.[1]),
        };
      }

      return { changes: 1 };
    });

    const result = await recordFeedback({
      transactionId: 'transaction-id',
      finalCategoryId: 'power-id',
      prediction,
    });

    expect(result?.feedback_status).toBe('accepted_top1');
    expect(sendFeedback).toHaveBeenCalledWith(
      expect.objectContaining({
        appliedCategoryId: 'Utilities & Services',
        predictedCategoryId: 'Utilities & Services',
      }),
    );
  });
});
