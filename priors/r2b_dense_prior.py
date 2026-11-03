"""R2b: does the votes-before-query prior also lift DENSE retrieval?

Vectors : the multilingual-e5-base embeddings of the pipeline's embedding step (F3). Arm A = full ticket text at 256 tokens, arm B = title + question at 512.
Scores  : recomputed here on CPU as exact cosine from the saved vectors (the saved rankings carry no scores and no dev split).
Prior   : score = (s - s_100) / (s_1 - s_100) inside the top 100  +  alpha * log(1 + net votes before the query's day).
          The min-max scaling matters: e5 cosines sit in a narrow band (about 0.7 to 0.9), so a ratio like BM25's would barely move.
Protocol: alpha is tuned on the same 1,000-query dev slice as R2 (seed 21); held-out (1,079 queries) is run ONCE.
Usage   : python r2b_dense_prior.py   -> results_dense_prior.json, outputs_dense_prior/*.npy
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from eval_utils import paired, summarize      # noqa: E402
from votes_asof import asof, build             # noqa: E402

SF = Path(os.environ.get("DATA_DIR", ""))        # folder with gold/, parquet/, xml/ and the embedding folders (see README)
GOLD = SF / "gold"
ARMS = (("e5 full text @256", "emb_e5base"), ("e5 title+question @512", "emb_e5base_tq512"))
ALPHAS = (0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

# ---------------------------------------------------------------------------------------------------------------------------
# Data: the ticket order, the gold queries, the dev slice and the held-out set
# ---------------------------------------------------------------------------------------------------------------------------
ticket_ids = np.array(pq.read_table(GOLD / "tickets.parquet", columns=["id"]).to_pydict()["id"])
position_of = {int(ticket_id): row for row, ticket_id in enumerate(ticket_ids)}
votes_before = build(ticket_ids)

all_queries = [json.loads(line) for split in ("dev", "heldout") for line in open(GOLD / f"gold_dup_{split}.jsonl")]
rng = np.random.default_rng(21)
dev_all = [q for q in all_queries if q["split"] == "dev"]
dev = [dev_all[i] for i in rng.choice(len(dev_all), 1000, replace=False)]        # the same slice R2 and R3 tuned on
held_out = [q for q in all_queries if q["split"] == "heldout"]


def load_vectors(path):
    table = pq.read_table(path)
    ids = np.array(table["id"].to_pylist())
    vectors = np.stack(table["vec"].to_numpy(zero_copy_only=False)).astype(np.float32)
    return ids, vectors


def top_k(ticket_vectors, query_vectors, k=100, batch=256):
    """Exact cosine top-k (vectors are normalized, so the dot product is the cosine): (row indices, scores), best first."""
    all_idx, all_scores = [], []
    for start in range(0, len(query_vectors), batch):
        scores = query_vectors[start:start + batch] @ ticket_vectors.T
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, 1)
        order = np.argsort(-top_scores, 1)
        all_idx.append(np.take_along_axis(top, order, 1))
        all_scores.append(np.take_along_axis(top_scores, order, 1))
    return np.vstack(all_idx), np.vstack(all_scores)


def add_votes_prior(idx, scores, queries, alpha):
    if alpha == 0:
        return idx

    lowest, highest = scores[:, -1:], scores[:, :1]
    scaled = (scores - lowest) / np.maximum(highest - lowest, 1e-9)

    known_votes = np.array(
        [[asof(votes_before, ticket_ids[j], q["created"][:10]) for j in row] for row, q in zip(idx, queries)],
        dtype=np.float32,
    )
    final = scaled + alpha * np.log1p(known_votes)
    order = np.argsort(-final, 1)
    return np.take_along_axis(idx, order, 1)


def first_hits(rankings, queries):
    """1-based rank of the first original inside the top 10 for each query (None = miss)."""
    hits = []
    for ranking, q in zip(rankings, queries):
        originals = {position_of[o] for o in q["originals"] if o in position_of}
        hits.append(next((rank for rank, doc in enumerate(ranking[:10], 1) if doc in originals), None))
    return hits


def metrics(rankings, queries):
    hits = first_hits(rankings, queries)
    return summarize(hits), hits


# ---------------------------------------------------------------------------------------------------------------------------
# One pass per embedding variant: tune alpha on dev, then score held-out once
# ---------------------------------------------------------------------------------------------------------------------------
out = {"tuned_on": "1000 dev queries (seed 21); held-out run once", "arms": {}, "dev_grid": {}}
out_dir = HERE / "outputs_dense_prior"
out_dir.mkdir(exist_ok=True)

for name, folder in ARMS:
    started = time.time()
    ticket_row_ids, ticket_vectors = load_vectors(SF / folder / "tickets.parquet")
    assert (ticket_row_ids == ticket_ids).all(), "embedding rows must follow the ticket order"

    query_row_ids, query_vectors = load_vectors(SF / folder / "q_clean.parquet")
    query_row = {int(i): j for j, i in enumerate(query_row_ids)}
    vectors_for = lambda queries: query_vectors[[query_row[q["qid"]] for q in queries]]

    # dev: pick the prior weight with the best MRR@10
    dev_idx, dev_scores = top_k(ticket_vectors, vectors_for(dev))
    dev_grid = {}
    for alpha in ALPHAS:
        m, _ = metrics(add_votes_prior(dev_idx, dev_scores, dev, alpha), dev)
        dev_grid[alpha] = m
        print(name, "dev alpha", alpha, m, flush=True)
    best_alpha = max(dev_grid, key=lambda a: dev_grid[a]["MRR10"])
    out["dev_grid"][name] = {str(a): m for a, m in dev_grid.items()}

    # held-out, once: plain dense, then dense + prior with the dev-chosen alpha
    held_idx, held_scores = top_k(ticket_vectors, vectors_for(held_out))
    dense_metrics, dense_hits = metrics(held_idx, held_out)
    with_prior = add_votes_prior(held_idx, held_scores, held_out, best_alpha)
    prior_metrics, prior_hits = metrics(with_prior, held_out)

    prior_name = f"{name}: dense + votes prior (alpha {best_alpha}, tuned on dev)"
    out["arms"][f"{name}: dense"] = dense_metrics
    out["arms"][prior_name] = {**prior_metrics, "alpha": best_alpha, "paired_vs_dense": paired(prior_hits, dense_hits)}
    print(name, "HELD-OUT dense", dense_metrics, "| +prior", prior_metrics, "| paired", out["arms"][prior_name]["paired_vs_dense"], round(time.time() - started), "s", flush=True)

    tag = "fulltext256" if "full" in name else "titleq512"
    np.save(out_dir / f"{tag}_dense.npy", ticket_ids[held_idx])
    np.save(out_dir / f"{tag}_dense_prior.npy", ticket_ids[with_prior])

    # each matrix is about 1 GB: free it before the next variant is loaded
    del ticket_vectors, query_vectors, ticket_row_ids
    gc.collect()

json.dump(out, open(HERE / "results_dense_prior.json", "w"), indent=1)
