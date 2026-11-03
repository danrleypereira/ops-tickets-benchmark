"""Two tests on the serverfault duplicate benchmark, e5 vectors (full ticket at 256 tokens), exact cosine, clean queries.

(a) Tag filter. A new ticket comes with its tags, so we can keep only tickets that share at least one tag with it.
    Does that help? Recall@10 with and without the filter, paired bootstrap interval, and how often the filter
    throws the right answer away.

(b) "No confident match". We cannot ask a person to label "no answer exists", so we simulate it:
    answerable   = the query, with its original in the ticket base
    unanswerable = the same query with its original(s) removed
    Confidence = the similarity of the top hit. A threshold is chosen on the dev queries (best balanced accuracy)
    and applied once to the held-out queries. AUROC = how well the confidence separates the two cases (0.5 = chance).

Input : SF_DIR (e5 vectors, tickets, posts, gold files). Output: OUT_JSON. RAM about 2.5 GB, CPU only.

usage: f9_filter_abstain.py SF_DIR OUT_JSON
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import torch

sf_dir, out_path = sys.argv[1:3]
MASKED = -1e9   # score given to tickets that are excluded, so they can never reach the top


# ----------------------------------------------------------------------------- ticket vectors
ids, parts = [], []
for batch in pq.ParquetFile(os.path.join(sf_dir, 'emb_e5base/tickets.parquet')).iter_batches(batch_size=20000):
    data = batch.to_pydict()
    ids += data['id']
    parts.append(np.array(data['vec'], np.float32))
ticket_vectors = torch.from_numpy(np.concatenate(parts))
ticket_ids = np.array(ids)
position_of = {int(i): j for j, i in enumerate(ticket_ids)}


# ----------------------------------------------------------------------------- tags
ticket_tags = {}
for batch in pq.ParquetFile(os.path.join(sf_dir, 'gold/tickets.parquet')).iter_batches(columns=['id', 'tags'], batch_size=50000):
    data = batch.to_pydict()
    for ticket_id, tags in zip(data['id'], data['tags']):
        ticket_tags[ticket_id] = set(tags.split())

queries = []
for part in ('dev', 'heldout'):
    with open(os.path.join(sf_dir, f'gold/gold_dup_{part}.jsonl')) as fh:
        queries += [json.loads(line) for line in fh]

# The queries were removed from the ticket base (they are closed duplicates), so their tags come from the raw posts.
wanted = {q['qid'] for q in queries}
query_tags = {}
for batch in pq.ParquetFile(os.path.join(sf_dir, 'parquet/posts.parquet')).iter_batches(columns=['id', 'tags'], batch_size=50000):
    data = batch.to_pydict()
    for post_id, tags in zip(data['id'], data['tags']):
        if post_id in wanted:
            query_tags[post_id] = set(t for t in (tags or '').replace('><', '|').strip('<>').split('|') if t)

# inverted index: tag -> positions of the tickets that carry it
tickets_with_tag = defaultdict(list)
for position, tag_id in enumerate(ticket_ids):
    for tag in ticket_tags[int(tag_id)]:
        tickets_with_tag[tag].append(position)

# ----------------------------------------------------------------------------- query vectors
query_table = pq.read_table(os.path.join(sf_dir, 'emb_e5base/q_clean.parquet'))
query_vectors = np.stack(query_table['vec'].to_numpy(zero_copy_only=False)).astype(np.float32)
query_row = {i: j for j, i in enumerate(query_table['id'].to_pylist())}


def rank_of_right_ticket(scores, right_positions, k=10):
    """1-based rank of the first right ticket in the top k, or None."""
    top = torch.topk(scores, k).indices.numpy()
    for rank, position in enumerate(top, 1):
        if position in right_positions:
            return rank
    return None


# ----------------------------------------------------------------------------- run every query
per_query = {'dev': [], 'heldout': []}

for q in queries:
    scores = ticket_vectors @ torch.from_numpy(query_vectors[query_row[q['qid']]])
    right = {position_of[o] for o in q['originals']}

    # (a) plain search, then search restricted to tickets that share a tag with the query
    plain_rank = rank_of_right_ticket(scores, right)
    candidates = sorted({p for tag in query_tags.get(q['qid'], ()) for p in tickets_with_tag.get(tag, ())})
    if candidates:
        filtered_scores = torch.full_like(scores, MASKED)
        keep = torch.tensor(candidates)
        filtered_scores[keep] = scores[keep]
        filtered_rank = rank_of_right_ticket(filtered_scores, right)
    else:
        filtered_rank = plain_rank

    # (b) top similarity with the original present, and with the original removed
    top_with_original = float(scores.max())
    without_original = scores.clone()
    for position in right:
        without_original[position] = MASKED

    per_query[q['split']].append(dict(
        base=plain_rank, filt=filtered_rank, ncand=len(candidates),
        top1_full=top_with_original, top1_masked=float(without_original.max()),
        recall_in_filter=bool(right & set(candidates)),
    ))


def recall10_flags(ranks):
    return np.array([1.0 if r and r <= 10 else 0.0 for r in ranks])


results = {}
heldout = per_query['heldout']

# ----------------------------------------------------------------------------- (a) result
filtered = recall10_flags([x['filt'] for x in heldout])
unfiltered = recall10_flags([x['base'] for x in heldout])
difference = filtered - unfiltered

rng = np.random.default_rng(13)
means = [difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(10000)]
results['tag_filter'] = dict(
    n=len(heldout), R10_unfiltered=round(float(unfiltered.mean()), 4), R10_filtered=round(float(filtered.mean()), 4), diff=round(float(difference.mean()), 4),
    ci95=[round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
    median_candidates=int(np.median([x['ncand'] for x in heldout])),
    original_survives_filter_pct=round(100 * float(np.mean([x['recall_in_filter'] for x in heldout])), 1),
)


# ----------------------------------------------------------------------------- (b) result
def auroc(positive, negative):
    """Chance that a random answerable query has a higher confidence than a random unanswerable one."""
    both = np.concatenate([positive, negative])
    ranks = both.argsort().argsort() + 1
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


dev_answerable = np.array([x['top1_full'] for x in per_query['dev']])
dev_unanswerable = np.array([x['top1_masked'] for x in per_query['dev']])

# threshold with the best balanced accuracy on dev (answer when confidence >= threshold)
best_accuracy, threshold = max(
    ((np.mean(dev_answerable >= t) + np.mean(dev_unanswerable < t)) / 2, t) for t in np.linspace(0.6, 1.0, 401)
)

answerable = np.array([x['top1_full'] for x in heldout])
unanswerable = np.array([x['top1_masked'] for x in heldout])
results['abstention'] = dict(
    threshold_from_dev=round(float(threshold), 4), dev_balanced_accuracy=round(float(best_accuracy), 4),
    heldout_auroc=round(auroc(answerable, unanswerable), 4),
    heldout_answerable_answered_pct=round(100 * float(np.mean(answerable >= threshold)), 1),
    heldout_unanswerable_abstained_pct=round(100 * float(np.mean(unanswerable < threshold)), 1),
    heldout_mean_top1_answerable=round(float(answerable.mean()), 4), heldout_mean_top1_unanswerable=round(float(unanswerable.mean()), 4),
)

with open(out_path, 'w') as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
