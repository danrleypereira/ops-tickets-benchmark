"""F7 baselines for the two-hop questions: BM25, dense (e5, full ticket at 256 tokens) and their fusion, WITHOUT using links.

They show what the link-following arms of f7_graph.py have to beat. Same metric: path recall@5 = share of the
2 needed tickets found in the top 5; "full path" = both found.

Input : SF_DIR (tickets and e5 vectors from the ops benchmark) and one or more question files (jsonl).
Output: OUT_JSON with a summary per split and arm and the top-10 per question.
RAM about 6-8 GB because the BM25 matrix is built over all tickets; CPU only. Announce RAM use before running.

usage: f7_bm25.py SF_DIR OUT_JSON GOLD1.jsonl [GOLD2.jsonl ...]
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

sf_dir, out_path, *gold_paths = sys.argv[1:]

BM25_K1, BM25_B = 1.5, 0.75
RRF_K = 60
TOP = 100

questions = []
for path in gold_paths:
    with open(path) as fh:
        questions += [json.loads(line) for line in fh]

# ----------------------------------------------------------------------------- BM25 over all tickets (same formula as f4_eval.py)
tickets = pq.read_table(os.path.join(sf_dir, 'gold/tickets.parquet'), columns=['id', 'ticket_text']).to_pydict()
ticket_ids = np.array(tickets['id'])

counter = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w\w+\b', dtype=np.float32)
counts = counter.fit_transform(tickets['ticket_text']).tocsr()
del tickets   # the text is large; only the counts are needed from here

n_docs = counts.shape[0]
doc_length = np.asarray(counts.sum(1)).ravel()
avg_length = doc_length.mean()
docs_with_term = np.asarray((counts > 0).sum(0)).ravel()
idf = np.log((n_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1).astype(np.float32)

cells = counts.tocoo()
tf_part = cells.data * (BM25_K1 + 1) / (cells.data + BM25_K1 * (1 - BM25_B + BM25_B * doc_length[cells.row] / avg_length))
bm25_weights = sp.csr_matrix((tf_part * idf[cells.col], (cells.row, cells.col)), shape=cells.shape).T.tocsr()
del counts, cells

query_terms = counter.transform([q['q'] for q in questions])
query_terms.data[:] = 1                          # a term counts once, however often the query repeats it
scores = query_terms.tocsr() @ bm25_weights      # questions x tickets
bm25_ranking = [np.argsort(-scores.getrow(i).toarray().ravel())[:TOP] for i in range(len(questions))]
del bm25_weights, scores

# ----------------------------------------------------------------------------- dense search with the stored e5 vectors
vector_ids, vector_parts = [], []
for batch in pq.ParquetFile(os.path.join(sf_dir, 'emb_e5base/tickets.parquet')).iter_batches(batch_size=20000):
    data = batch.to_pydict()
    vector_ids += data['id']
    vector_parts.append(np.array(data['vec'], np.float32))
assert (np.array(vector_ids) == ticket_ids).all(), 'vector file and ticket file are not in the same order'
ticket_vectors = torch.from_numpy(np.concatenate(vector_parts))
del vector_parts

encoder = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
encoder.max_seq_length = 256
question_vectors = encoder.encode(['query: ' + q['q'] for q in questions], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
dense_ranking = [(ticket_vectors @ torch.from_numpy(v)).numpy().argsort()[::-1][:TOP] for v in question_vectors]


def rrf(first, second):
    """Reciprocal rank fusion: each ticket scores 1 / (k + rank) in each list it appears in."""
    score = {}
    for rank, doc in enumerate(first):
        score[doc] = score.get(doc, 0) + 1 / (RRF_K + rank + 1)
    for rank, doc in enumerate(second):
        score[doc] = score.get(doc, 0) + 1 / (RRF_K + rank + 1)
    return [doc for doc, _ in sorted(score.items(), key=lambda item: -item[1])[:TOP]]


# ----------------------------------------------------------------------------- score
per_question = []
for question, bm25, dense in zip(questions, bm25_ranking, dense_ranking):
    needed = set(question['relevant'])
    row = dict(qid=question['qid'], split=question['split'])

    for arm, ranking in (('bm25', bm25), ('dense', dense), ('rrf', rrf(list(dense), list(bm25)))):
        top = [int(ticket_ids[i]) for i in ranking[:10]]
        row[arm] = dict(
            r5=len(needed & set(top[:5])) / len(needed),
            r10=len(needed & set(top)) / len(needed),
            full5=int(needed <= set(top[:5])),
            rank=top,
        )
    per_question.append(row)

summary = {}
for split in sorted({row['split'] for row in per_question}):
    rows = [row for row in per_question if row['split'] == split]
    for arm in ('bm25', 'dense', 'rrf'):
        summary[f'{split}/{arm}'] = dict(
            n=len(rows),
            path_recall5=round(float(np.mean([row[arm]['r5'] for row in rows])), 3),
            full_paths5=int(sum(row[arm]['full5'] for row in rows)),
            recall10=round(float(np.mean([row[arm]['r10'] for row in rows])), 3),
        )

with open(out_path, 'w') as fh:
    json.dump(dict(summary=summary, per_question=per_question), fh, indent=1)
for key, value in summary.items():
    print(key, value)
