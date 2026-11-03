"""Paired comparison of two dense arms on the same held-out queries, from their saved top-100 rankings.

Why paired: both arms answer the same 1,079 questions, so the fair test is on the per-question difference (resampling the
questions), not on two separate scores. If the 95% interval of the difference excludes 0, the difference is not chance.

Input : two f4_eval.py output folders (clean_dense.npy = top-100 ticket ids per query) and the held-out gold file.
Output: OUT_JSON with arm B minus arm A for Recall@1, Recall@10 and MRR@10, each with a bootstrap interval.

usage: f17_paired_arms.py GOLD_HELDOUT_JSONL OUTPUTS_DIR_A OUTPUTS_DIR_B OUT_JSON     (B minus A)
"""
import json
import os
import sys

import numpy as np

gold_path, dir_a, dir_b, out_path = sys.argv[1:5]
RESAMPLES, SEED = 10000, 13

with open(gold_path) as fh:
    queries = [json.loads(line) for line in fh]


def first_hits(outputs_dir):
    """Rank (1-based) of the first right ticket within the top 10 for every query, or None."""
    rankings = np.load(os.path.join(outputs_dir, 'clean_dense.npy'))
    hits = []
    for ranked, query in zip(rankings, queries):
        right = set(query['originals'])
        hits.append(next((place for place, ticket in enumerate(ranked[:10].tolist(), 1) if ticket in right), None))
    return hits


hits_a, hits_b = first_hits(dir_a), first_hits(dir_b)

measures = {
    'R1': lambda hits: np.array([1.0 if h == 1 else 0.0 for h in hits]),
    'R10': lambda hits: np.array([1.0 if h else 0.0 for h in hits]),
    'MRR10': lambda hits: np.array([1 / h if h else 0.0 for h in hits]),
}

rng = np.random.default_rng(SEED)
n = len(queries)
results = {'queries': n, 'a': dir_a, 'b': dir_b, 'b_minus_a': {}}
for name, measure in measures.items():
    difference = measure(hits_b) - measure(hits_a)
    means = [difference[rng.integers(0, n, n)].mean() for _ in range(RESAMPLES)]
    results['b_minus_a'][name] = dict(
        a=round(float(measure(hits_a).mean()), 4), b=round(float(measure(hits_b).mean()), 4), diff=round(float(difference.mean()), 4),
        ci95=[round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
    )
    print(name, results['b_minus_a'][name])

with open(out_path, 'w') as fh:
    json.dump(results, fh, indent=1)
