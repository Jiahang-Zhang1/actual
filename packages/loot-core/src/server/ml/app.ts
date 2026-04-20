import { createApp } from '#server/app';
import { mutator } from '#server/mutators';

import { predictCategory } from './service';
import { getLatestPrediction, recordFeedback, savePrediction } from './store';
import type { MlPredictRequest } from './types';

export type MlHandlers = {
  'ml-predict-category': typeof predictAndPersist;
  'ml-get-latest-prediction': typeof getPredictionForTransaction;
  'ml-record-feedback': typeof saveFeedback;
};

export const app = createApp<MlHandlers>();

app.method('ml-predict-category', mutator(predictAndPersist));
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

async function getPredictionForTransaction(transactionId: string) {
  return await getLatestPrediction(transactionId);
}

async function saveFeedback(args: {
  transactionId: string;
  finalCategoryId: string;
}) {
  return await recordFeedback(args);
}
