import { logger } from '#platform/server/log';
import { createApp } from '#server/app';
import { aqlQuery } from '#server/aql';
import * as db from '#server/db';
import { predictCategory, predictCategoryBatch } from '#server/ml/service';
import { savePrediction } from '#server/ml/store';
import type { MlPredictRequest } from '#server/ml/types';
import { mutator } from '#server/mutators';
import { undoable } from '#server/undo';
import { q, Query } from '#shared/query';
import type { QueryState } from '#shared/query';
import type {
  AccountEntity,
  CategoryGroupEntity,
  PayeeEntity,
  TransactionEntity,
} from '#types/models';

import { exportQueryToCSV, exportToCSV } from './export/export-to-csv';
import { parseFile } from './import/parse-file';
import type { ParseFileOptions } from './import/parse-file';
import { mergeTransactions } from './merge';

import { batchUpdateTransactions } from '.';

export type TransactionHandlers = {
  'transactions-batch-update': typeof handleBatchUpdateTransactions;
  'transaction-add': typeof addTransaction;
  'transaction-update': typeof updateTransaction;
  'transaction-delete': typeof deleteTransaction;
  'transaction-move': typeof moveTransaction;
  'transactions-parse-file': typeof parseTransactionsFile;
  'transactions-export': typeof exportTransactions;
  'transactions-export-query': typeof exportTransactionsQuery;
  'transactions-merge': typeof mergeTransactions;
  'get-earliest-transaction': typeof getEarliestTransaction;
  'get-latest-transaction': typeof getLatestTransaction;
};

type PredictionCandidate = {
  transactionId: string;
  payload: MlPredictRequest;
};

async function handleBatchUpdateTransactions({
  added,
  deleted,
  updated,
  learnCategories,
  runTransfers = true,
}: Parameters<typeof batchUpdateTransactions>[0]) {
  const result = await batchUpdateTransactions({
    added,
    updated,
    deleted,
    learnCategories,
    runTransfers,
  });

  const candidates = [...(added || []), ...(updated || [])];
  await maybePredictForTransactions(candidates);

  return result;
}

async function addTransaction(transaction: TransactionEntity) {
  await handleBatchUpdateTransactions({ added: [transaction] });
  return {};
}

async function updateTransaction(transaction: TransactionEntity) {
  await handleBatchUpdateTransactions({ updated: [transaction] });
  return {};
}

async function deleteTransaction(transaction: Pick<TransactionEntity, 'id'>) {
  await handleBatchUpdateTransactions({ deleted: [transaction] });
  return {};
}

async function moveTransaction({
  id,
  accountId,
  targetId,
}: {
  id: string;
  accountId: string;
  targetId: string | null;
}) {
  // Fetch the transaction to validate it exists and verify account
  const transaction = await db.getTransaction(id);
  if (!transaction) {
    throw new Error(`Transaction not found: ${id}`);
  }

  // Validate that the provided accountId matches the transaction's actual account
  // This prevents sort order calculations against the wrong account
  if (transaction.account !== accountId) {
    throw new Error(
      `Account mismatch: transaction belongs to account ${transaction.account}, not ${accountId}`,
    );
  }

  // Child transactions can be reordered within their parent's children
  // The db.moveTransaction handles the sibling-scoped reordering for children

  await db.moveTransaction(id, accountId, targetId);
  return {};
}

async function parseTransactionsFile({
  filepath,
  options,
}: {
  filepath: string;
  options: ParseFileOptions;
}) {
  return parseFile(filepath, options);
}

async function exportTransactions({
  transactions,
  accounts,
  categoryGroups,
  payees,
}: {
  transactions: TransactionEntity[];
  accounts: AccountEntity[];
  categoryGroups: CategoryGroupEntity[];
  payees: PayeeEntity[];
}) {
  return exportToCSV(transactions, accounts, categoryGroups, payees);
}

async function exportTransactionsQuery({
  query: queryState,
}: {
  query: QueryState;
}) {
  return exportQueryToCSV(new Query(queryState));
}

async function getEarliestTransaction() {
  const { data } = await aqlQuery(
    q('transactions')
      .options({ splits: 'none' })
      .orderBy({ date: 'asc' })
      .select('*')
      .limit(1),
  );
  return data[0] || null;
}

async function getLatestTransaction() {
  const { data } = await aqlQuery(
    q('transactions')
      .options({ splits: 'none' })
      .orderBy({ date: 'desc' })
      .select('*')
      .limit(1),
  );
  return data[0] || null;
}

async function toPredictionCandidate(
  transactionRef: Partial<TransactionEntity> | undefined,
): Promise<null | PredictionCandidate> {
  if (!transactionRef?.id) {
    return null;
  }

  const transaction = await db.getTransaction(transactionRef.id);
  if (!transaction || transaction.is_child || transaction.is_parent) {
    return null;
  }

  // Only generate suggestions for uncategorized transactions to avoid
  // overwriting explicit user categorization decisions.
  if (transaction.category) {
    return null;
  }

  const payeeName =
    transaction.payee != null
      ? (await db.getPayee(transaction.payee))?.name
      : null;
  const accountName =
    transaction.account != null
      ? (await db.getAccount(transaction.account))?.name
      : null;

  const description =
    payeeName ||
    transaction.imported_payee ||
    transaction.notes ||
    accountName ||
    'manual entry';

  const payload: MlPredictRequest = {
    transactionId: transaction.id,
    transactionDescription: description,
    country: 'unknown',
    currency: 'unknown',
    amount: typeof transaction.amount === 'number' ? transaction.amount : null,
    transactionDate: transaction.date || null,
    accountId: transaction.account || null,
    notes: transaction.notes || null,
    importedDescription: transaction.imported_payee || null,
  };

  return {
    transactionId: transaction.id,
    payload,
  };
}

async function maybePredictForTransaction(args: {
  transactionId: string;
  payload: MlPredictRequest;
}) {
  try {
    const prediction = await predictCategory(args.payload);
    if (!prediction) {
      return null;
    }

    await savePrediction(args.transactionId, prediction);
    return prediction;
  } catch (err) {
    logger.error('ML prediction failed', err);
    return null;
  }
}

async function maybePredictForTransactions(
  transactions: Array<Partial<TransactionEntity> | undefined>,
) {
  const candidates = (
    await Promise.all(
      transactions.map(transaction => toPredictionCandidate(transaction)),
    )
  ).filter((candidate): candidate is PredictionCandidate => candidate !== null);

  if (candidates.length === 0) {
    return;
  }

  try {
    // Use serving batch inference for throughput and consistent monitoring
    // over multi-transaction updates/import flows.
    const predictions = await predictCategoryBatch(
      candidates.map(candidate => candidate.payload),
    );
    await Promise.all(
      predictions.map(async (prediction, index) => {
        if (!prediction) {
          return;
        }
        await savePrediction(candidates[index].transactionId, prediction);
      }),
    );
  } catch (err) {
    logger.error(
      'ML batch prediction failed; falling back to single prediction',
      err,
    );
    // Keep graceful degradation: if batch endpoint is unavailable, preserve
    // existing behavior by retrying item-by-item.
    await Promise.all(
      candidates.map(candidate => maybePredictForTransaction(candidate)),
    );
  }
}

export const app = createApp<TransactionHandlers>();

app.method(
  'transactions-batch-update',
  mutator(undoable(handleBatchUpdateTransactions)),
);
app.method('transactions-merge', mutator(undoable(mergeTransactions)));

app.method('transaction-add', mutator(addTransaction));
app.method('transaction-update', mutator(updateTransaction));
app.method('transaction-delete', mutator(deleteTransaction));
app.method('transaction-move', mutator(undoable(moveTransaction)));
app.method('transactions-parse-file', mutator(parseTransactionsFile));
app.method('transactions-export', mutator(exportTransactions));
app.method('transactions-export-query', mutator(exportTransactionsQuery));
app.method('get-earliest-transaction', getEarliestTransaction);
app.method('get-latest-transaction', getLatestTransaction);
