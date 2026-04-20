import type {
  MlBatchPredictResponse,
  MlFeedbackRequest,
  MlFeedbackResponse,
  MlPredictRequest,
  MlPredictResponse,
} from './types';

const ML_SERVICE_URL =
  process.env.ACTUAL_ML_SERVICE_URL || 'http://localhost:8000';
const ML_REQUEST_TIMEOUT_MS = Number(
  process.env.ACTUAL_ML_SERVICE_TIMEOUT_MS || 2500,
);

function sanitizeText(value?: null | string, maxLength: number = 512) {
  return (value ?? '').replace(/\s+/g, ' ').trim().slice(0, maxLength);
}

function toRequestBody(payload: MlPredictRequest) {
  return {
    transaction_id: payload.transactionId,
    transaction_description: sanitizeText(payload.transactionDescription),
    country: sanitizeText(payload.country, 32) || 'unknown',
    currency: sanitizeText(payload.currency, 16) || 'unknown',
    amount: payload.amount ?? null,
    transaction_date: payload.transactionDate ?? null,
    account_id: payload.accountId ?? null,
    notes: sanitizeText(payload.notes, 256) || null,
    imported_description:
      sanitizeText(payload.importedDescription, 256) || null,
  };
}

function parsePredictResponse(response: unknown): MlPredictResponse {
  const parsed = response as MlPredictResponse;
  if (!parsed || !Array.isArray(parsed.top_categories)) {
    throw new Error('ML predict response is missing top_categories');
  }
  return parsed;
}

function toFeedbackBody(payload: MlFeedbackRequest) {
  return {
    transaction_id: payload.transactionId,
    model_version: payload.modelVersion,
    predicted_category_id: payload.predictedCategoryId,
    applied_category_id: payload.appliedCategoryId,
    confidence: payload.confidence ?? null,
    candidate_category_ids: payload.candidateCategoryIds ?? [],
  };
}

async function fetchWithTimeout(url: string, init?: RequestInit) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), ML_REQUEST_TIMEOUT_MS);

  try {
    return await fetch(url, {
      ...init,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
}

export async function predictCategory(
  payload: MlPredictRequest,
): Promise<MlPredictResponse | null> {
  const body = toRequestBody(payload);
  if (!body.transaction_description) {
    return null;
  }

  const res = await fetchWithTimeout(`${ML_SERVICE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    throw new Error(`ML predict failed: ${res.status}`);
  }

  return parsePredictResponse(await res.json());
}

export async function predictCategoryBatch(
  payloads: MlPredictRequest[],
): Promise<Array<MlPredictResponse | null>> {
  if (payloads.length === 0) {
    return [];
  }

  const requestBodies = payloads.map(toRequestBody);
  const indexedValidBodies = requestBodies
    .map((body, index) => ({ body, index }))
    .filter(({ body }) => Boolean(body.transaction_description));

  const predictions: Array<MlPredictResponse | null> = payloads.map(() => null);
  if (indexedValidBodies.length === 0) {
    return predictions;
  }

  const res = await fetchWithTimeout(`${ML_SERVICE_URL}/predict_batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      items: indexedValidBodies.map(({ body }) => body),
    }),
  });

  if (!res.ok) {
    throw new Error(`ML predict batch failed: ${res.status}`);
  }

  const data = (await res.json()) as MlBatchPredictResponse;
  if (
    !Array.isArray(data.items) ||
    data.items.length !== indexedValidBodies.length
  ) {
    throw new Error(
      'ML predict batch response size does not match request size',
    );
  }

  data.items.forEach((item, itemIndex) => {
    const originalIndex = indexedValidBodies[itemIndex].index;
    predictions[originalIndex] = parsePredictResponse(item);
  });

  return predictions;
}

export async function mlHealth() {
  const healthzRes = await fetchWithTimeout(`${ML_SERVICE_URL}/healthz`);
  if (healthzRes.ok) {
    return healthzRes.json();
  }

  const healthRes = await fetchWithTimeout(`${ML_SERVICE_URL}/health`);
  if (!healthRes.ok) {
    throw new Error(
      `ML health failed: healthz=${healthzRes.status}, health=${healthRes.status}`,
    );
  }

  return healthRes.json();
}

export async function sendFeedback(
  payload: MlFeedbackRequest,
): Promise<MlFeedbackResponse> {
  const res = await fetchWithTimeout(`${ML_SERVICE_URL}/feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(toFeedbackBody(payload)),
  });

  if (!res.ok) {
    throw new Error(`ML feedback failed: ${res.status}`);
  }

  return (await res.json()) as MlFeedbackResponse;
}
