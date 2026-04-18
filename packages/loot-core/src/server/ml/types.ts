export type MlPredictRequest = {
  transactionId?: string;
  transactionDescription: string;
  country?: string | null;
  currency?: string | null;
};

export type MlCategoryScore = {
  category_id: string;
  score: number;
};

export type MlPredictResponse = {
  predicted_category_id: string;
  confidence: number;
  top_categories: MlCategoryScore[];
  model_version: string;
};

export type MlPredictionRecord = {
  id?: string;
  transaction_id: string;
  model_version: string;
  predicted_category_id: string;
  confidence: number;
  top_categories_json: string;
  status: 'pending' | 'accepted_top1' | 'accepted_top3' | 'overridden' | 'ignored';
  created_at: string;
  updated_at: string;
};
