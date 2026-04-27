import type {
  MlBatchPredictResponse,
  MlCategoryScore,
  MlFeedbackRequest,
  MlFeedbackResponse,
  MlPredictRequest,
  MlPredictResponse,
} from './types';

const ML_SERVICE_URL = getMlServiceUrl();
const ML_REQUEST_TIMEOUT_MS = Number(
  process.env.ACTUAL_ML_SERVICE_TIMEOUT_MS || 2500,
);

function getMlServiceUrl() {
  const explicitUrl = (process.env.ACTUAL_ML_SERVICE_URL || '').trim();
  if (explicitUrl) {
    return explicitUrl.replace(/\/+$/, '');
  }

  const isBrowserLike =
    typeof window !== 'undefined' || typeof self !== 'undefined';
  const isDevelopment = process.env.NODE_ENV === 'development';

  if (isBrowserLike && !isDevelopment) {
    // The compiled backend worker runs inside the user's browser tab. In
    // production we need a same-origin HTTPS path so requests are not blocked
    // by mixed-content rules or internal cluster DNS names.
    return '/smartcat';
  }

  return 'http://localhost:8000';
}

// Normalize user-provided text fields before sending to serving to avoid
// oversized payloads and incidental whitespace variance.
function sanitizeText(value?: null | string, maxLength: number = 512) {
  return (value ?? '').replace(/\s+/g, ' ').trim().slice(0, maxLength);
}

function buildFallbackDescription(payload: MlPredictRequest) {
  const candidates = [
    sanitizeText(payload.transactionDescription),
    sanitizeText(payload.importedDescription, 256),
    sanitizeText(payload.notes, 256),
  ].filter(Boolean);

  if (candidates.length > 0) {
    return candidates[0];
  }

  const hints = [
    payload.accountId ? `account ${sanitizeText(payload.accountId, 64)}` : '',
    payload.currency && sanitizeText(payload.currency, 16) !== 'unknown'
      ? sanitizeText(payload.currency, 16)
      : '',
    typeof payload.amount === 'number' && Number.isFinite(payload.amount)
      ? `amount ${Math.abs(payload.amount)}`
      : '',
  ].filter(Boolean);

  return hints.join(' ').trim() || 'manual entry';
}

function toRequestBody(payload: MlPredictRequest) {
  return {
    transaction_id: payload.transactionId,
    transaction_description: buildFallbackDescription(payload),
    merchant_text: sanitizeText(payload.transactionDescription, 256) || null,
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

// Guard the HTTP contract so callers always receive a top-k shape that the
// frontend badge/popover logic can render safely.
function normalizeTopCategories(topCategories: unknown): MlCategoryScore[] {
  if (!Array.isArray(topCategories)) {
    return [];
  }

  return topCategories
    .filter(
      (
        item,
      ): item is {
        category_id: string;
        score: number;
      } =>
        !!item &&
        typeof item === 'object' &&
        'category_id' in item &&
        typeof item.category_id === 'string' &&
        'score' in item &&
        typeof item.score === 'number',
    )
    .sort((left, right) => right.score - left.score)
    .slice(0, 3);
}

function parsePredictResponse(response: unknown): MlPredictResponse {
  const parsed = response as Partial<MlPredictResponse> | null;
  const topCategories = normalizeTopCategories(parsed?.top_categories);

  if (!parsed || typeof parsed.model_version !== 'string') {
    throw new Error('ML predict response is missing model_version');
  }
  if (topCategories.length === 0) {
    throw new Error('ML predict response is missing top_categories');
  }

  const topCategory = topCategories[0];
  return {
    predicted_category_id: topCategory.category_id,
    confidence: topCategory.score,
    top_categories: topCategories,
    model_version: parsed.model_version,
  };
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
  const indexedBodies = requestBodies.map((body, index) => ({ body, index }));

  // Keep output cardinality equal to input cardinality so transaction-level
  // callers can map predictions back by original index.
  const predictions: Array<MlPredictResponse | null> = payloads.map(() => null);

  const res = await fetchWithTimeout(`${ML_SERVICE_URL}/predict_batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      items: indexedBodies.map(({ body }) => body),
    }),
  });

  if (!res.ok) {
    throw new Error(`ML predict batch failed: ${res.status}`);
  }

  const data = (await res.json()) as MlBatchPredictResponse;
  if (
    !Array.isArray(data.items) ||
    data.items.length !== indexedBodies.length
  ) {
    throw new Error(
      'ML predict batch response size does not match request size',
    );
  }

  // The serving endpoint returns only valid items. Rehydrate sparse results
  // into the original request order expected by callers.
  data.items.forEach((item, itemIndex) => {
    const originalIndex = indexedBodies[itemIndex].index;
    predictions[originalIndex] = parsePredictResponse(item);
  });

  return predictions;
}

export async function mlHealth() {
  // Support both modern (/healthz) and legacy (/health) serving images.
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
