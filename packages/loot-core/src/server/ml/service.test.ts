import { afterEach, describe, expect, it, vi } from 'vitest';

import { predictCategory, predictCategoryBatch } from './service';

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
    },
  });
}

describe('ml service request shaping', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('predictCategory sends a fallback description for sparse manual entries', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({
          predicted_category_id: 'Food & Dining',
          confidence: 0.91,
          top_categories: [
            { category_id: 'Food & Dining', score: 0.61 },
            { category_id: 'Shopping & Retail', score: 0.25 },
            { category_id: 'Transportation', score: 0.14 },
          ],
          model_version: 'v-test',
        }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const response = await predictCategory({
      transactionDescription: '',
      accountId: 'checking-account',
      currency: 'USD',
      amount: 24.99,
    });

    expect(response).toMatchObject({
      predicted_category_id: 'Food & Dining',
      confidence: 0.61,
    });

    const [, init] = fetchMock.mock.calls[0];
    const body = JSON.parse(String(init?.body));
    expect(body.transaction_description).toContain('account checking-account');
    expect(fetchMock.mock.calls[0][0]).toBe('http://localhost:8000/predict');
  });

  it('predictCategoryBatch preserves sparse rows and normalizes confidence to top-1', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({
          items: [
            {
              predicted_category_id: 'Shopping & Retail',
              confidence: 0.97,
              top_categories: [
                { category_id: 'Food & Dining', score: 0.58 },
                { category_id: 'Shopping & Retail', score: 0.31 },
                { category_id: 'Transportation', score: 0.11 },
              ],
              model_version: 'v-batch',
            },
          ],
        }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const responses = await predictCategoryBatch([
      {
        transactionDescription: '',
        notes: '',
        accountId: 'checking-account',
        amount: 18.2,
      },
    ]);

    expect(responses).toHaveLength(1);
    expect(responses[0]).toMatchObject({
      predicted_category_id: 'Food & Dining',
      confidence: 0.58,
      top_categories: [
        { category_id: 'Food & Dining', score: 0.58 },
        { category_id: 'Shopping & Retail', score: 0.31 },
        { category_id: 'Transportation', score: 0.11 },
      ],
    });
  });

  it('predictCategory uses the same-origin smartcat proxy in browser production builds', async () => {
    const previousNodeEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'production';
    try {
      const fetchMock = vi
        .fn()
        .mockResolvedValue(
          jsonResponse({
            predicted_category_id: 'Food & Dining',
            confidence: 0.9,
            top_categories: [
              { category_id: 'Food & Dining', score: 0.6 },
              { category_id: 'Shopping & Retail', score: 0.25 },
              { category_id: 'Transportation', score: 0.15 },
            ],
            model_version: 'v-browser',
          }),
        );

      vi.resetModules();
      vi.stubGlobal('fetch', fetchMock);
      vi.stubGlobal('self', {});

      const { predictCategory: predictCategoryInBrowser } = await import(
        './service'
      );

      await predictCategoryInBrowser({
        transactionDescription: 'STARBUCKS',
        amount: 5.75,
        currency: 'USD',
      });

      expect(fetchMock.mock.calls[0][0]).toBe('/smartcat/predict');
    } finally {
      process.env.NODE_ENV = previousNodeEnv;
    }
  });
});
