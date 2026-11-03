"""Does a popularity signal also lift bge-m3? (serverfault duplicate benchmark)

This applies the votes-prior method (first used with the e5 vectors) to the
bge-m3 vectors, with the same dev slice (1,000 dev queries, seed 21), the same score scaling and the same rule of running
the held-out queries only once.

The signal: the net votes a ticket had BEFORE the day of the query (so no information from the future). Moderators close a
new question against the established, popular question, so popular tickets are more likely to be the right answer.

Final score of a candidate = (similarity rescaled inside the top 100 to 0..1) + alpha * log(1 + votes before the query day).
alpha is chosen on the dev slice by MRR@10; the chosen alpha is then applied once to the held-out queries.

Needs the lxml package (votes_asof.py parses Votes.xml): run with the environment that has it, for example
python -I pipeline/f14_bge_prior.py

Input : $OPS_BENCH_DIR/sf (tickets, gold files, emb_bgem3 vectors).
Output: results/bgem3_dense_prior.json and results/bgem3_prior_outputs/*.npy (top-100 ticket ids per held-out query).
"""
import gc
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from votes_asof import build, asof   # vote history: net votes of a ticket as of a given day (needs lxml)

SF = Path(os.environ.get('OPS_BENCH_DIR', Path.home() / 'ops-benchmark')) / 'sf'
GOLD = SF / 'gold'
OUT_JSON = SF / 'results/bgem3_dense_prior.json'
OUTPUTS = SF / 'results/bgem3_prior_outputs'
OUTPUTS.mkdir(exist_ok=True)

ALPHAS = (0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)   # 0 = no prior (plain dense)
TOP = 100

# ----------------------------------------------------------------------------- tickets, votes, queries
ticket_ids = np.array(pq.read_table(GOLD / 'tickets.parquet', columns=['id']).to_pydict()['id'])
position_of = {int(i): j for j, i in enumerate(ticket_ids)}
vote_history = build(ticket_ids)

all_queries = []
for part in ('dev', 'heldout'):
    with open(GOLD / f'gold_dup_{part}.jsonl') as fh:
        all_queries += [json.loads(line) for line in fh]

dev_queries_all = [q for q in all_queries if q['split'] == 'dev']
dev_slice_rng = np.random.default_rng(21)   # the same dev slice as the e5 runs
dev = [dev_queries_all[i] for i in dev_slice_rng.choice(len(dev_queries_all), 1000, replace=False)]
heldout = [q for q in all_queries if q['split'] == 'heldout']


def read_vectors(path):
    table = pq.read_table(path)
    ids = np.array(table['id'].to_pylist())
    vectors = np.stack(table['vec'].to_numpy(zero_copy_only=False)).astype(np.float32)
    return ids, vectors


def top_hits(ticket_vectors, query_vectors, k=TOP, batch=256):
    """Top-k ticket positions and their similarities per query, best first (exact search, in batches)."""
    positions, similarities = [], []
    for start in range(0, len(query_vectors), batch):
        scores = query_vectors[start:start + batch] @ ticket_vectors.T
        best = np.argpartition(-scores, k, axis=1)[:, :k]             # the k best, unordered
        best_scores = np.take_along_axis(scores, best, 1)
        order = np.argsort(-best_scores, 1)                           # ... then ordered
        positions.append(np.take_along_axis(best, order, 1))
        similarities.append(np.take_along_axis(best_scores, order, 1))
    return np.vstack(positions), np.vstack(similarities)


def rerank_with_votes(positions, similarities, queries, alpha):
    """Re-order each query's top hits by similarity plus the votes bonus."""
    if alpha == 0:
        return positions

    worst, best = similarities[:, -1:], similarities[:, :1]
    rescaled = (similarities - worst) / np.maximum(best - worst, 1e-9)   # 0..1 inside this query's top hits: cosines sit in a narrow band

    votes = np.array([[asof(vote_history, ticket_ids[p], q['created'][:10]) for p in row] for row, q in zip(positions, queries)], dtype=np.float32)
    final_score = rescaled + alpha * np.log1p(votes)

    order = np.argsort(-final_score, 1)
    return np.take_along_axis(positions, order, 1)


def metrics(ranking, queries):
    """Recall@1/5/10 and MRR@10, plus the rank of the first right ticket per query (None if outside the top 10)."""
    first_hit = []
    for ranked, q in zip(ranking, queries):
        right = {position_of[o] for o in q['originals'] if o in position_of}
        first_hit.append(next((place for place, p in enumerate(ranked[:10], 1) if p in right), None))

    n = len(first_hit)
    recall_within = lambda limit: round(sum(1 for h in first_hit if h and h <= limit) / n, 4)
    scores = dict(n=n, R1=recall_within(1), R5=recall_within(5), R10=recall_within(10),
                  MRR10=round(sum(1 / h for h in first_hit if h) / n, 4))
    return scores, first_hit


def paired_difference(hits_a, hits_b, resamples=10000, seed=13):
    """A minus B on the same queries with a bootstrap interval; if it excludes 0 the difference is not chance."""
    rng = np.random.default_rng(seed)
    n = len(hits_a)
    recall_diff = np.array([(1.0 if a and a <= 10 else 0.0) - (1.0 if b and b <= 10 else 0.0) for a, b in zip(hits_a, hits_b)])
    mrr_diff = np.array([(1 / a if a else 0.0) - (1 / b if b else 0.0) for a, b in zip(hits_a, hits_b)])

    result = {}
    for name, diff in (('R10', recall_diff), ('MRR10', mrr_diff)):
        means = diff[rng.integers(0, n, (resamples, n))].mean(1)
        result[name] = dict(diff=round(float(diff.mean()), 4),
                            ci95=[round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)])
    return result


# ----------------------------------------------------------------------------- bge-m3: tune alpha on dev, score held-out once
results = {'tuned_on': '1000 dev queries (seed 21); held-out run once', 'arms': {}, 'dev_grid': {}}
name = 'bge-m3 full text @256'
started = time.time()

vector_ids, ticket_vectors = read_vectors(SF / 'emb_bgem3/tickets.parquet')
assert (vector_ids == ticket_ids).all(), 'vector file and ticket file are not in the same order'
query_ids, query_vectors = read_vectors(SF / 'emb_bgem3/q_clean.parquet')
query_row = {int(i): j for j, i in enumerate(query_ids)}


def vectors_of(queries):
    return query_vectors[[query_row[q['qid']] for q in queries]]


# tuning on the dev slice
dev_positions, dev_similarities = top_hits(ticket_vectors, vectors_of(dev))
dev_grid = {}
for alpha in ALPHAS:
    scores, _ = metrics(rerank_with_votes(dev_positions, dev_similarities, dev, alpha), dev)
    dev_grid[alpha] = scores
    print(name, 'dev alpha', alpha, scores, flush=True)

best_alpha = max(dev_grid, key=lambda a: dev_grid[a]['MRR10'])
results['dev_grid'][name] = {str(a): scores for a, scores in dev_grid.items()}

# held-out, once
held_positions, held_similarities = top_hits(ticket_vectors, vectors_of(heldout))
dense_scores, dense_hits = metrics(held_positions, heldout)
with_prior = rerank_with_votes(held_positions, held_similarities, heldout, best_alpha)
prior_scores, prior_hits = metrics(with_prior, heldout)

prior_key = f'{name}: dense + votes prior (alpha {best_alpha}, tuned on dev)'
results['arms'][f'{name}: dense'] = dense_scores
results['arms'][prior_key] = {**prior_scores, 'alpha': best_alpha, 'paired_vs_dense': paired_difference(prior_hits, dense_hits)}
print(name, 'HELD-OUT dense', dense_scores, '| +prior', prior_scores, '| paired', results['arms'][prior_key]['paired_vs_dense'], round(time.time() - started), 's', flush=True)

np.save(OUTPUTS / 'bgem3_fulltext256_dense.npy', ticket_ids[held_positions])
np.save(OUTPUTS / 'bgem3_fulltext256_dense_prior.npy', ticket_ids[with_prior])
del ticket_vectors, query_vectors, vector_ids
gc.collect()

with open(OUT_JSON, 'w') as fh:
    json.dump(results, fh, indent=1)
