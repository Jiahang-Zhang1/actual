import type { MlPredictRequest, MlPredictResponse } from './types';

const ML_SERVICE_URL =
  process.env.ACTUAL_ML_SERVICE_URL || 'http://127.0.0.1:8000';

export async function predictCategory(
  payload: MlPredictRequest,
): Promise<MlPredictResponse | null> {
  const res = await fetch(`${ML_SERVICE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      transaction_id: payload.transactionId,
      transaction_description: payload.transactionDescription,
      country: payload.country ?? 'unknown',
      currency: payload.currency ?? 'unknown',
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`ML predict failed: ${res.status} ${text}`);
  }

  return (await res.json()) as MlPredictResponse;
}

export async function mlHealth() {
  const res = await fetch(`${ML_SERVICE_URL}/health`);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`ML health failed: ${res.status} ${text}`);
  }
  return res.json();
}