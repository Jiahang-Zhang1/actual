import { logger } from '#platform/server/log';
import { createApp } from '#server/app';
import { mutator } from '#server/mutators';

import { predictCategory, predictCategoryBatch } from './service';
import { getLatestPrediction, recordFeedback, savePrediction } from './store';
import type { MlPredictRequest, MlPredictResponse } from './types';

type BatchPredictionResult = {
  transactionId: string;
  prediction: MlPredictResponse | null;
};

export type MlHandlers = {
  'ml-predict-category': typeof predictAndPersist;
  'ml-predict-category-batch': typeof predictAndPersistBatch;
  'ml-get-latest-prediction': typeof getPredictionForTransaction;
  'ml-record-feedback': typeof saveFeedback;
};

export const app = createApp<MlHandlers>();

app.method('ml-predict-category', mutator(predictAndPersist));
app.method('ml-predict-category-batch', mutator(predictAndPersistBatch));
app.method('ml-get-latest-prediction', getPredictionForTransaction);
app.method('ml-record-feedback', mutator(saveFeedback));

async function predictAndPersist(args: {
  transactionId: string;
  payload: MlPredictRequest;
}) {
  const prediction = await predictCategory(args.payload);
  if (!prediction) {
    return null;
  }

  await savePrediction(args.transactionId, prediction);
  return prediction;
}

async function predictAndPersistBatch(args: {
  items: Array<{
    transactionId: string;
    payload: MlPredictRequest;
  }>;
}): Promise<BatchPredictionResult[]> {
  const items = (args.items || []).filter(
    item => item?.transactionId && item?.payload,
  );
  if (items.length === 0) {
    return [];
  }

  let predictions: Array<MlPredictResponse | null>;

  try {
    predictions = await predictCategoryBatch(items.map(item => item.payload));
  } catch (error) {
    logger.error(
      'ML batch prediction failed in ml app; falling back to single',
      error,
    );
    predictions = await Promise.all(
      items.map(async item => {
        try {
          return await predictCategory(item.payload);
        } catch (singleError) {
          logger.error('ML single prediction fallback failed', singleError);
          return null;
        }
      }),
    );
  }

  await Promise.all(
    predictions.map(async (prediction, index) => {
      if (!prediction) {
        return;
      }
      await savePrediction(items[index].transactionId, prediction);
    }),
  );

  return items.map((item, index) => ({
    transactionId: item.transactionId,
    prediction: predictions[index] ?? null,
  }));
}

async function getPredictionForTransaction(transactionId: string) {
  return await getLatestPrediction(transactionId);
}

async function saveFeedback(args: {
  transactionId: string;
  finalCategoryId: string;
  prediction?: MlPredictResponse | null;
  syncToServing?: boolean;
}) {
  return await recordFeedback(args);
}
