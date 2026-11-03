"""Can any similarity-based score tell "a matching ticket exists" from "none exists"? (serverfault, e5 vectors, clean queries)

Follow-up to f9_filter_abstain.py, which tried only the top similarity. Four candidate confidence signals:
  top1   similarity of the best hit
  gap    best hit minus second hit
  gap10  best hit minus the mean of hits 2..10
  z      how many standard deviations the best hit sits above the whole corpus (computed on the corpus without the removed originals)
Same simulation as f9: "answerable" = the original is in the base, "unanswerable" = the same query with its original(s) removed.
For each signal a threshold is chosen on the dev queries and applied once to the held-out queries; AUROC 0.5 means no separation.

A first version of the z signal scored a perfect 1.0. That was a bug: the removed original had been replaced by a huge negative number,
which distorted the mean and spread. The original is now dropped from the statistics instead.

Input : SF_DIR (e5 vectors, tickets, posts, gold files). Output: OUT_JSON. RAM about 2.5 GB, CPU only.

usage: f13_abstain_signals.py SF_DIR OUT_JSON
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sf_dir, out_path = sys.argv[1:3]

# ----------------------------------------------------------------------------- vectors and queries
ids, parts = [], []
for batch in pq.ParquetFile(os.path.join(sf_dir, 'emb_e5base/tickets.parquet')).iter_batches(batch_size=20000):
    data = batch.to_pydict()
    ids += data['id']
    parts.append(np.array(data['vec'], np.float32))
ticket_vectors = torch.from_numpy(np.concatenate(parts))
position_of = {int(i): j for j, i in enumerate(np.array(ids))}

queries = []
for part in ('dev', 'heldout'):
    with open(os.path.join(sf_dir, f'gold/gold_dup_{part}.jsonl')) as fh:
        queries += [json.loads(line) for line in fh]

query_table = pq.read_table(os.path.join(sf_dir, 'emb_e5base/q_clean.parquet'))
query_vectors = np.stack(query_table['vec'].to_numpy(zero_copy_only=False)).astype(np.float32)
query_row = {i: j for j, i in enumerate(query_table['id'].to_pylist())}


# ----------------------------------------------------------------------------- the four signals
def signals(scores, removed=()):
    """Confidence signals of one query. `removed` = positions taken out of the corpus (the original, in the unanswerable case)."""
    keep = torch.ones_like(scores, dtype=torch.bool)
    for position in removed:
        keep[position] = False
    kept = scores[keep]

    top10 = torch.topk(kept, 10).values.numpy()
    return dict(
        top1=float(top10[0]),
        gap=float(top10[0] - top10[1]),
        gap10=float(top10[0] - top10[1:].mean()),
        z=float((top10[0] - float(kept.mean())) / float(kept.std())),
    )


per_split = {'dev': [], 'heldout': []}   # (signals with the original present, signals with it removed) per query
for q in queries:
    scores = ticket_vectors @ torch.from_numpy(query_vectors[query_row[q['qid']]])
    right = {position_of[o] for o in q['originals']}
    per_split[q['split']].append((signals(scores), signals(scores, right)))


def auroc(positive, negative):
    """Chance that a random answerable query scores higher than a random unanswerable one."""
    both = np.concatenate([positive, negative])
    ranks = both.argsort().argsort() + 1
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


# ----------------------------------------------------------------------------- threshold on dev, one scoring on held-out
results = {}
for name in ('top1', 'gap', 'gap10', 'z'):
    dev_answerable = np.array([pair[0][name] for pair in per_split['dev']])
    dev_unanswerable = np.array([pair[1][name] for pair in per_split['dev']])
    held_answerable = np.array([pair[0][name] for pair in per_split['heldout']])
    held_unanswerable = np.array([pair[1][name] for pair in per_split['heldout']])

    # candidate thresholds: 197 quantiles of all dev values; answer when the signal is at or above the threshold
    candidates = np.quantile(np.concatenate([dev_answerable, dev_unanswerable]), np.linspace(0.01, 0.99, 197))
    best_accuracy, threshold = max(
        ((np.mean(dev_answerable >= t) + np.mean(dev_unanswerable < t)) / 2, t) for t in candidates
    )

    results[name] = dict(
        dev_balanced_accuracy=round(float(best_accuracy), 4), threshold=round(float(threshold), 4),
        heldout_auroc=round(auroc(held_answerable, held_unanswerable), 4),
        heldout_answerable_answered_pct=round(100 * float(np.mean(held_answerable >= threshold)), 1),
        heldout_unanswerable_abstained_pct=round(100 * float(np.mean(held_unanswerable < threshold)), 1),
    )

with open(out_path, 'w') as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
