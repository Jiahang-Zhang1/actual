import { createApp } from '#server/app';
import { aqlQuery } from '#server/aql';
import * as db from '#server/db';
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

import { predictCategory } from '#server/ml/service';
import { savePrediction } from '#server/ml/store';

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

  for (const trans of candidates) {
    if (trans?.id) {
      const savedTransaction = await db.getTransaction(trans.id);
      if (savedTransaction) {
        await maybePredictForTransaction(savedTransaction as TransactionEntity);
      }
    }
  }

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


async function getPayeeName(payeeId?: string | null) {
  if (!payeeId) {
    return '';
  }

  const { data } = await aqlQuery(
    q('payees')
      .filter({ id: payeeId })
      .select('name')
      .limit(1),
  );

  return ((data?.[0] as { name?: string } | undefined)?.name ?? '').trim();
}


async function maybePredictForTransaction(transaction: TransactionEntity) {
  const payeeName = await getPayeeName(transaction.payee);
  const description = (
    payeeName ||
    transaction.imported_payee ||
    transaction.notes ||
    ''
  ).trim();

  console.log('ML candidate transaction:', {
    id: transaction.id,
    payee: transaction.payee,
    payeeName,
    importedPayee: transaction.imported_payee,
    notes: transaction.notes,
    description,
  });

  if (!description) {
    console.log('ML skipped: empty description', transaction.id);
    return null;
  }

  try {
    const prediction = await predictCategory({
      transactionId: transaction.id,
      transactionDescription: description,
      country: 'unknown',
      currency: transaction.currency || 'unknown',
    });

    console.log('ML prediction result:', transaction.id, prediction);

    if (!prediction) {
      return null;
    }

    if (prediction.confidence < 0.35) {
      console.log(
        'ML skipped: low confidence',
        transaction.id,
        prediction.confidence,
      );
      return null;
    }

    await savePrediction(transaction.id, prediction);
    console.log('ML prediction saved:', transaction.id);

    return prediction;
  } catch (err) {
    console.error('ML prediction failed', transaction.id, err);
    return null;
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
