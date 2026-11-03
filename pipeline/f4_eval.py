"""F4: score BM25, dense search and their fusion on the duplicate gold set.

For each gold query the system must put the original ticket in the top 10. Reported: Recall@1/5/10 and MRR@10,
a bootstrap interval for Recall@10, and paired differences between arms (the same queries are used for every arm,
so the differences are tested on matching pairs, not on two separate samples).

Every query is scored twice: "clean" (banner and links to the original removed) and "raw" (text as found in the dump).
The gap between the two is the leakage.

Input : tickets.parquet and gold_dup_*.jsonl in GOLD_DIR; vectors in EMB_DIR (made by f3_embed.py), or "none" for BM25 only.
Output: OUT_JSON with the numbers, and OUT_JSON minus ".json" + "_outputs/" with the top-100 ticket ids per query and arm
        (so every score can be recomputed later without searching again).

BM25 is a sparse-matrix version (Lucene-style idf, k1 = 1.5, b = 0.75), because the rank_bm25 library is too slow for 320k
documents. It follows the formula of the scratch version on the Python and SQL page, not rank_bm25's idf floor.

usage: f4_eval.py GOLD_DIR EMB_DIR OUT_JSON [split=heldout]
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer

gold, emb, out_path = sys.argv[1:4]
split = sys.argv[4] if len(sys.argv) > 4 else 'heldout'
dense_enabled = emb != 'none'

BM25_K1, BM25_B = 1.5, 0.75
RRF_K = 60            # standard constant of reciprocal rank fusion: damps the weight of the very top ranks
TOP = 100             # ranking depth kept for fusion and for the saved outputs


# ----------------------------------------------------------------------------- load
tickets = pq.read_table(os.path.join(gold, 'tickets.parquet'), columns=['id', 'ticket_text']).to_pydict()
ticket_ids = np.array(tickets['id'])
position_of = {int(i): j for j, i in enumerate(ticket_ids)}

queries = []
for part in ('dev', 'heldout'):
    with open(os.path.join(gold, f'gold_dup_{part}.jsonl')) as fh:
        queries += [json.loads(line) for line in fh]
queries = [q for q in queries if q['split'] == split]
relevant = [{position_of[o] for o in q['originals']} for q in queries]


def read_vectors(name):
    table = pq.read_table(os.path.join(emb, name + '.parquet'))
    ids = np.array(table['id'].to_pylist())
    vectors = np.stack(table['vec'].to_numpy(zero_copy_only=False)).astype(np.float32)
    return ids, vectors


ticket_vectors = None
if dense_enabled:
    vector_ids, vectors = read_vectors('tickets')
    assert (vector_ids == ticket_ids).all(), 'vector file and ticket file are not in the same order'
    ticket_vectors = torch.from_numpy(vectors)


# ----------------------------------------------------------------------------- dense search (exact cosine)
def dense_rank(query_vectors):
    """Top-100 ticket positions per query. Vectors are unit length, so a dot product is the cosine similarity."""
    ranked = []
    for start in range(0, len(query_vectors), 128):   # in batches, to keep the score matrix small
        scores = torch.from_numpy(query_vectors[start:start + 128]) @ ticket_vectors.T
        ranked.append(scores.topk(TOP, dim=1).indices.numpy())
    return np.concatenate(ranked)


# ----------------------------------------------------------------------------- BM25 (sparse)
counter = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w\w+\b', max_df=1.0, dtype=np.float32)
counts = counter.fit_transform(tickets['ticket_text']).tocsr()          # documents x terms

n_docs = counts.shape[0]
doc_length = np.asarray(counts.sum(1)).ravel()
avg_length = doc_length.mean()
docs_with_term = np.asarray((counts > 0).sum(0)).ravel()
idf = np.log((n_docs - docs_with_term + 0.5) / (docs_with_term + 0.5) + 1).astype(np.float32)   # Lucene form: never negative

# weight of each (document, term) cell: saturating term frequency, normalised by document length, times idf
cells = counts.tocoo()
tf_part = cells.data * (BM25_K1 + 1) / (cells.data + BM25_K1 * (1 - BM25_B + BM25_B * doc_length[cells.row] / avg_length))
bm25_weights = sp.csr_matrix((tf_part * idf[cells.col], (cells.row, cells.col)), shape=cells.shape).T.tocsr()   # terms x documents


def bm25_rank(query_texts):
    query_terms = counter.transform(query_texts)
    query_terms.data[:] = 1                       # a term counts once, however often the query repeats it
    scores = query_terms.tocsr() @ bm25_weights   # queries x documents

    ranked = []
    for i in range(scores.shape[0]):
        row = scores.getrow(i).toarray().ravel()
        best = np.argpartition(-row, TOP)[:TOP]               # the 100 best, unordered
        ranked.append(best[np.argsort(-row[best])])           # ... then ordered
    return np.stack(ranked)


# ----------------------------------------------------------------------------- fusion
def rrf(first, second):
    """Reciprocal rank fusion: each ticket scores 1 / (k + rank) in every list it appears in; add them up."""
    fused = []
    for a, b in zip(first, second):
        score = {}
        for rank, doc in enumerate(a):
            score[doc] = score.get(doc, 0) + 1 / (RRF_K + rank + 1)
        for rank, doc in enumerate(b):
            score[doc] = score.get(doc, 0) + 1 / (RRF_K + rank + 1)
        best = sorted(score.items(), key=lambda item: -item[1])[:TOP]
        fused.append([doc for doc, _ in best])
    return np.array(fused)


# ----------------------------------------------------------------------------- metrics
def metrics(ranking):
    """Recall@1/5/10 and MRR@10. Also returns, per query, the rank of the first right ticket (None if outside the top 10)."""
    first_hit = []
    for ranked, right in zip(ranking, relevant):
        hit = next((place for place, doc in enumerate(ranked[:10], 1) if doc in right), None)
        first_hit.append(hit)

    n = len(relevant)
    at1 = sum(h == 1 for h in first_hit)
    at5 = sum(bool(h and h <= 5) for h in first_hit)
    at10 = sum(h is not None for h in first_hit)
    mrr = sum(1 / h for h in first_hit if h)
    scores = dict(n=n, R1=round(at1 / n, 4), R5=round(at5 / n, 4), R10=round(at10 / n, 4), MRR10=round(mrr / n, 4))
    return scores, first_hit


def bootstrap_recall10(first_hit, resamples=2000, seed=13):
    """95% interval for Recall@10: resample the queries with replacement and look at the spread of the mean."""
    rng = np.random.default_rng(seed)
    hit10 = np.array([1.0 if h and h <= 10 else 0.0 for h in first_hit])
    n = len(hit10)
    means = [hit10[rng.integers(0, n, n)].mean() for _ in range(resamples)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def paired_difference(hits_a, hits_b, resamples=10000, seed=13):
    """arm A minus arm B on the same queries, with a bootstrap interval. If the interval excludes 0 the difference is not chance."""
    rng = np.random.default_rng(seed)
    n = len(hits_a)
    recall_diff = np.array([(1.0 if a and a <= 10 else 0.0) - (1.0 if b and b <= 10 else 0.0) for a, b in zip(hits_a, hits_b)])
    mrr_diff = np.array([(1 / a if a else 0.0) - (1 / b if b else 0.0) for a, b in zip(hits_a, hits_b)])

    result = {}
    for name, diff in (('R10', recall_diff), ('MRR10', mrr_diff)):
        draws = rng.integers(0, n, (resamples, n))
        means = diff[draws].mean(1)
        result[name] = dict(diff=round(float(diff.mean()), 4),
                            ci95=[round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)])
    return result


# ----------------------------------------------------------------------------- run every arm on clean and raw queries
results = {'split': split, 'n': len(queries)}
first_hits = {}
outputs_dir = os.path.splitext(out_path)[0] + '_outputs'
os.makedirs(outputs_dir, exist_ok=True)

for variant, vector_file, text_field in (('clean', 'q_clean', 'query_clean'), ('raw', 'q_raw', 'query_raw')):
    bm25_ranking = bm25_rank([q[text_field] for q in queries])
    arms = [('bm25', bm25_ranking)]

    if dense_enabled:
        query_ids, query_vectors = read_vectors(vector_file)
        row_of = {i: j for j, i in enumerate(query_ids)}
        query_vectors = query_vectors[[row_of[q['qid']] for q in queries]]   # same order as the gold queries

        dense_ranking = dense_rank(query_vectors)
        arms = [('dense', dense_ranking), ('bm25', bm25_ranking), ('rrf', rrf(dense_ranking, bm25_ranking))]

    for arm, ranking in arms:
        scores, first_hit = metrics(ranking)
        scores['R10_ci95'] = bootstrap_recall10(first_hit)
        results[f'{variant}/{arm}'] = scores
        print(variant, arm, scores, flush=True)

        first_hits[(variant, arm)] = first_hit
        np.save(os.path.join(outputs_dir, f'{variant}_{arm}.npy'), ticket_ids[ranking[:, :TOP]])

# ----------------------------------------------------------------------------- paired comparisons
if dense_enabled:
    for variant in ('clean', 'raw'):
        results[f'paired/{variant}/dense-bm25'] = paired_difference(first_hits[(variant, 'dense')], first_hits[(variant, 'bm25')])
        results[f'paired/{variant}/rrf-dense'] = paired_difference(first_hits[(variant, 'rrf')], first_hits[(variant, 'dense')])
    for key, value in results.items():
        if key.startswith('paired'):
            print(key, value, flush=True)

# leakage = the same arm scored on raw queries minus clean queries
for arm in (['dense', 'bm25', 'rrf'] if dense_enabled else ['bm25']):
    results[f'paired/leakage/{arm}(raw-clean)'] = paired_difference(first_hits[('raw', arm)], first_hits[('clean', arm)])
    print('leakage', arm, results[f'paired/leakage/{arm}(raw-clean)'], flush=True)

with open(out_path, 'w') as fh:
    json.dump(results, fh, indent=1)
