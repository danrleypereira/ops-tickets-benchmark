"""Metrics shared by the R2, R2b and R3 scripts, so every arm is scored the same way.

A "first hit" is the 1-based rank of the first relevant ticket inside the top 10 for one query, or None if it is absent.
"""
import numpy as np


def summarize(first_hits):
    """Recall@1/5/10 and MRR@10 from a list of first-hit ranks (None = a miss)."""
    n = len(first_hits)
    share = lambda keep: round(sum(1 for hit in first_hits if hit and keep(hit)) / n, 4)

    return dict(
        n=n,
        R1=share(lambda hit: hit == 1),
        R5=share(lambda hit: hit <= 5),
        R10=share(lambda hit: hit <= 10),
        MRR10=round(sum(1 / hit for hit in first_hits if hit) / n, 4),
    )


def paired(hits_a, hits_b, resamples=10000, seed=13):
    """Paired bootstrap of arm A minus arm B on the SAME queries.

    Resampling queries (not scores) keeps the pairing, so the interval answers "does A beat B on these questions"
    instead of comparing two separate noisy averages.
    """
    rng = np.random.default_rng(seed)
    n = len(hits_a)

    recall10_diff = np.array([(1.0 if a and a <= 10 else 0.0) - (1.0 if b and b <= 10 else 0.0) for a, b in zip(hits_a, hits_b)])
    mrr_diff = np.array([(1 / a if a else 0.0) - (1 / b if b else 0.0) for a, b in zip(hits_a, hits_b)])

    out = {}
    for name, diffs in (("R10", recall10_diff), ("MRR10", mrr_diff)):
        resampled_means = diffs[rng.integers(0, n, (resamples, n))].mean(1)
        out[name] = dict(
            diff=round(float(diffs.mean()), 4),
            ci95=[round(float(np.percentile(resampled_means, 2.5)), 4), round(float(np.percentile(resampled_means, 97.5)), 4)],
        )
    return out
