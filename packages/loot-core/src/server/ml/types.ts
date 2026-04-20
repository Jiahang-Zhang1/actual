export type MlPredictRequest = {
  transactionId?: string;
  transactionDescription: string;
  country?: string | null;
  currency?: string | null;
  amount?: number | null;
  transactionDate?: string | null;
  accountId?: string | null;
  notes?: string | null;
  importedDescription?: string | null;
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

export type MlBatchPredictResponse = {
  items: MlPredictResponse[];
};

export type MlFeedbackRequest = {
  transactionId: string;
  modelVersion: string;
  predictedCategoryId: string;
  appliedCategoryId: string;
  confidence?: null | number;
  candidateCategoryIds?: string[];
};

export type MlFeedbackResponse = {
  status: string;
  saved: boolean;
};

export type MlPredictionStatus =
  | 'pending'
  | 'accepted_top1'
  | 'accepted_top3'
  | 'overridden'
  | 'ignored';

export type MlPredictionRecord = {
  id?: string;
  transaction_id: string;
  model_version: string;
  predicted_category_id: string;
  confidence: number;
  top_categories_json: string;
  status: MlPredictionStatus;
  created_at: string;
  updated_at: string;
};
