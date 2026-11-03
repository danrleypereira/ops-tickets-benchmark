"""R2: query reformulation and a votes prior on top of sparse BM25.

Task   : 320,296 serverfault tickets. A gold query is a question that moderators closed as a duplicate; the target is its original.
Index  : the same sparse BM25 as the pipeline's f4_eval.py (Lucene idf, k1 1.5, b 0.75), so numbers are comparable.
Queries: the cleaned ones (`query_clean`), never the raw ones that still carry the banner naming the original.
Arms   : noise stripping, title weighting, error-code weighting, pseudo-relevance feedback (PRF), and a prior built from votes.

Protocol: tune every choice on a seeded 1,000-query slice of DEV, then run HELD-OUT once with the chosen configuration.
Usage   : python r2.py tune    -> r2_dev.json                (grid on dev)
          python r2.py final   -> results.json + outputs/    (held-out, once)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from eval_utils import paired, summarize
from votes_asof import asof as votes_asof
from votes_asof import build as build_votes

DATA = Path(os.environ.get("DATA_DIR", ""))        # folder with gold/ and parquet/ (see README)
GOLD = DATA / "gold"
POSTS = DATA / "parquet" / "posts.parquet"
HERE = Path(__file__).parent
MODE = sys.argv[1] if len(sys.argv) > 1 else "tune"

# ---------------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------------
tickets = pq.read_table(GOLD / "tickets.parquet", columns=["id", "ticket_text", "has_accepted", "created"]).to_pydict()
ticket_ids = np.array(tickets["id"])
position_of = {int(ticket_id): row for row, ticket_id in enumerate(ticket_ids)}      # ticket id -> row in the BM25 matrix
accepted = np.array(tickets["has_accepted"], dtype=np.float32)
created_at = np.array(tickets["created"], dtype="datetime64[ms]")

# `votes` is the score from the 2024 dump: it includes votes that arrived after the duplicate was closed (future information)
posts = pq.read_table(POSTS, columns=["id", "type", "score"]).to_pydict()
question_score = {pid: score for pid, kind, score in zip(posts["id"], posts["type"], posts["score"]) if kind == 1}
votes = np.array([max(question_score.get(int(i), 0) or 0, 0) for i in ticket_ids], dtype=np.float32)

votes_before = build_votes(ticket_ids)       # net votes per post as of any day, rebuilt from Votes.xml

all_queries = [json.loads(line) for split in ("dev", "heldout") for line in open(GOLD / f"gold_dup_{split}.jsonl")]

# ---------------------------------------------------------------------------------------------------------------------------
# BM25 index (sparse matrices, so 320k documents fit in memory and one query is a single matrix product)
# ---------------------------------------------------------------------------------------------------------------------------
vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b", dtype=np.float32)
counts = vectorizer.fit_transform(tickets["ticket_text"]).tocsr()
n_docs, vocab_size = counts.shape

doc_len = np.asarray(counts.sum(1)).ravel()
avg_len = doc_len.mean()
doc_freq = np.asarray((counts > 0).sum(0)).ravel()

# Lucene's idf (always positive) with k1 = 1.5, b = 0.75: the same parameters as the f4 harness
K1, B = 1.5, 0.75
idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1).astype(np.float32)

coo = counts.tocoo()
term_weight = coo.data * (K1 + 1) / (coo.data + K1 * (1 - B + B * doc_len[coo.row] / avg_len))
doc_term = sp.csr_matrix((term_weight * idf[coo.col], (coo.row, coo.col)), shape=counts.shape)    # (docs x terms), used by PRF
term_doc = doc_term.T.tocsr()                                                                       # (terms x docs), used to score queries
del coo, term_weight, counts

vocab = vectorizer.vocabulary_
analyze = vectorizer.build_analyzer()

# ---------------------------------------------------------------------------------------------------------------------------
# Query reformulation
# ---------------------------------------------------------------------------------------------------------------------------
# Machine noise that carries no meaning for a duplicate match: URLs, GUIDs, IPs, timestamps, Windows paths, unix paths, long hex, giant tokens
NOISE_PATTERNS = [re.compile(p, re.I) for p in (
    r"https?://\S+",
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    r"\b\d{1,3}(\.\d{1,3}){3}\b",
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?\S*",
    r"\b[a-z]:\\\S+",
    r"(?<!\w)/(?:[\w.-]+/){2,}[\w.-]*",
    r"\b[0-9a-f]{16,}\b",
    r"\b\w{26,}\b",
)]

# Error codes: hex values, vendor-style codes (ORA-12154) and numbers that follow words such as "error" or "event id"
ERROR_CODE = re.compile(r"0x[0-9a-f]{4,}|\b[a-z]{2,6}-\d{3,6}\b|(?:error|event id|eventid|code|status|errno|exit code)\W{0,4}(\d{2,10})", re.I)


def strip_noise(text):
    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def query_weights(text, noise=False, title_w=1.0, code_w=1.0):
    """{term index: weight}. Every query term starts at weight 1 (binary, as in the f4 harness); titles and error codes can be boosted."""
    title = text.partition("\n")[0]
    weights = {}

    for token in analyze(strip_noise(text) if noise else text):
        j = vocab.get(token)
        if j is not None:
            weights[j] = 1.0

    if title_w != 1.0:      # the first line of a question is its title: it names the problem more tightly than the body
        for token in analyze(strip_noise(title) if noise else title):
            j = vocab.get(token)
            if j is not None:
                weights[j] = max(weights.get(j, 0), title_w)

    if code_w != 1.0:
        for match in ERROR_CODE.finditer(text):
            for token in analyze(match.group(1) or match.group(0)):
                j = vocab.get(token)
                if j is not None:
                    weights[j] = max(weights.get(j, 0), code_w)

    return weights


def to_csr(weight_dicts):
    rows, cols, vals = [], [], []
    for row, weights in enumerate(weight_dicts):
        for j, value in weights.items():
            rows.append(row)
            cols.append(j)
            vals.append(value)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(weight_dicts), vocab_size), dtype=np.float32)


# ---------------------------------------------------------------------------------------------------------------------------
# Retrieval and the votes prior
# ---------------------------------------------------------------------------------------------------------------------------
def search(weight_dicts, k=100, prf=None, batch=100, cutoff=None):
    """BM25 top-k as (row indices, scores).

    prf    = (docs, terms, lam): Rocchio-style pseudo-relevance feedback. Average the BM25 weights of the top `docs` results,
             take the `terms` heaviest terms and add them to the query with weight `lam`.
    cutoff = per-query creation time; tickets created at or after it are masked (a question cannot duplicate a later one).
    """
    all_idx, all_scores = [], []

    for start in range(0, len(weight_dicts), batch):
        part = weight_dicts[start:start + batch]
        scores = (to_csr(part) @ term_doc).toarray()

        if cutoff is not None:
            scores[created_at[None, :] >= cutoff[start:start + batch, None]] = -1.0

        if prf:
            n_docs_fb, n_terms, lam = prf
            top_docs = np.argpartition(-scores, n_docs_fb, axis=1)[:, :n_docs_fb]
            expansions = []
            for r, docs in enumerate(top_docs):
                term_mass = np.asarray(doc_term[docs].sum(0)).ravel() / n_docs_fb
                best_terms = np.argpartition(-term_mass, n_terms)[:n_terms]
                scale = term_mass[best_terms].max() or 1.0
                expansions.append({int(j): lam * float(term_mass[j] / scale) for j in best_terms if int(j) not in part[r]})
            scores = scores + (to_csr(expansions) @ term_doc).toarray()

        # argpartition finds the top k without sorting 320k scores; only those k are then sorted
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, 1)
        order = np.argsort(-top_scores, 1)
        all_idx.append(np.take_along_axis(top, order, 1))
        all_scores.append(np.take_along_axis(top_scores, order, 1))

    return np.vstack(all_idx), np.vstack(all_scores)


def prior_rerank(idx, scores, kind, alpha, beta, queries=None):
    """Re-rank the BM25 top 100 with a popularity prior: bm25 / best_bm25 + alpha * log(1 + votes) (+ beta * accepted).

    kind 'dump': votes and the accepted flag from the 2024 dump. That is future information, so it is only an upper bound.
    kind 'asof': net votes cast strictly before the query's day, no accepted flag. This is what a live system could know.
    """
    relative = scores / np.maximum(scores[:, :1], 1e-9)

    if kind == "dump":
        final = relative + alpha * np.log1p(votes[idx]) + beta * accepted[idx]
    else:
        known_votes = np.array(
            [[votes_asof(votes_before, ticket_ids[j], q["created"][:10]) for j in row] for row, q in zip(idx, queries)],
            dtype=np.float32,
        )
        final = relative + alpha * np.log1p(known_votes)

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


def run(queries, cfg, prior=None):
    weights = [query_weights(q["query_clean"], cfg.get("noise", False), cfg.get("title_w", 1.0), cfg.get("code_w", 1.0)) for q in queries]
    cutoff = np.array([q["created"] for q in queries], dtype="datetime64[ms]") if cfg.get("asof") else None

    idx, scores = search(weights, prf=cfg.get("prf"), cutoff=cutoff)
    if prior:
        idx = prior_rerank(idx, scores, *prior, queries=queries)
    return idx


# ---------------------------------------------------------------------------------------------------------------------------
# tune: a grid on the dev slice
# ---------------------------------------------------------------------------------------------------------------------------
def tune():
    rng = np.random.default_rng(21)
    dev = [q for q in all_queries if q["split"] == "dev"]
    queries = [dev[i] for i in rng.choice(len(dev), 1000, replace=False)]
    out = {"dev_queries": len(queries), "seed": 21, "grid": []}

    def record(name, cfg, prior=None, legit=True):
        started = time.time()
        m, _ = metrics(run(queries, cfg, prior), queries)
        out["grid"].append({"name": name, "cfg": cfg, "prior": prior, "legit": legit, **m})
        print(name, m, round(time.time() - started), "s", flush=True)

    def best_without_prior():
        # the date filter is excluded on purpose: 13% of originals are newer than their duplicate, so it removes the right answer for them
        candidates = (g for g in out["grid"] if not g["cfg"].get("asof") and g["prior"] is None)
        return max(candidates, key=lambda g: g["MRR10"])

    record("bm25 clean (baseline)", {})
    record("bm25 + as-of date filter", dict(asof=True))
    for name, cfg in (("noise strip", dict(noise=True)),
                      ("noise + title x2", dict(noise=True, title_w=2.0)),
                      ("noise + title x3", dict(noise=True, title_w=3.0)),
                      ("noise + title x3 + code x3", dict(noise=True, title_w=3.0, code_w=3.0))):
        record(name, cfg)

    best_reform = best_without_prior()
    print("best reformulation:", best_reform["name"], flush=True)
    for docs, terms, lam in ((5, 20, 0.3), (10, 20, 0.3), (10, 50, 0.3), (5, 20, 0.6)):
        record(f"{best_reform['name']} + PRF docs={docs} terms={terms} lam={lam}", {**best_reform["cfg"], "prf": (docs, terms, lam)})

    best = best_without_prior()
    cfg = best["cfg"]
    print("best retrieval config:", best["name"], flush=True)

    # the prior grid reuses one search: only the re-ranking changes
    weights = [query_weights(q["query_clean"], cfg.get("noise", False), cfg.get("title_w", 1.0), cfg.get("code_w", 1.0)) for q in queries]
    idx, scores = search(weights, prf=cfg.get("prf"))

    prior_grids = (("asof", (0.02, 0.05, 0.1, 0.2, 0.4), (0,), True),
                   ("dump", (0.05, 0.1, 0.2), (0, 0.1), False))
    for kind, alphas, betas, legit in prior_grids:
        for alpha in alphas:
            for beta in betas:
                m, _ = metrics(prior_rerank(idx, scores, kind, alpha, beta, queries=queries), queries)
                out["grid"].append({"name": f"prior {kind} votes={alpha} accepted={beta} on [{best['name']}]", "cfg": cfg,
                                    "prior": [kind, alpha, beta], "legit": legit, **m})
                print("prior", kind, alpha, beta, m, flush=True)

    out["chosen"] = max((g for g in out["grid"] if g["legit"]), key=lambda g: g["MRR10"])
    out["best_leaky_upper_bound"] = max(out["grid"], key=lambda g: g["MRR10"])
    json.dump(out, open(HERE / "r2_dev.json", "w"), indent=1, default=list)
    print("CHOSEN (legit)", out["chosen"]["name"], flush=True)
    print("UPPER BOUND (dump-time votes)", out["best_leaky_upper_bound"]["name"], flush=True)


# ---------------------------------------------------------------------------------------------------------------------------
# final: held-out, once, with what dev chose (no tuning on held-out)
# ---------------------------------------------------------------------------------------------------------------------------
def final():
    dev = json.load(open(HERE / "r2_dev.json"))
    held_out = [q for q in all_queries if q["split"] == "heldout"]

    def to_cfg(saved):          # JSON turns the PRF tuple into a list
        return {k: (tuple(v) if k == "prf" else v) for k, v in saved.items()}

    def to_prior(saved):
        return tuple(saved) if saved else None

    reported = ("bm25 clean (baseline)", "bm25 + as-of date filter", "noise + title x3")
    arms = [(g["name"], to_cfg(g["cfg"]), to_prior(g["prior"])) for g in dev["grid"] if g["name"] in reported]

    chosen = dev["chosen"]
    arms.append(("CHOSEN on dev (legit): " + chosen["name"], to_cfg(chosen["cfg"]), to_prior(chosen["prior"])))
    bound = dev["best_leaky_upper_bound"]
    arms.append(("UPPER BOUND, uses dump-time votes: " + bound["name"], to_cfg(bound["cfg"]), to_prior(bound["prior"])))

    seen = set()
    arms = [arm for arm in arms if not (arm[0] in seen or seen.add(arm[0]))]       # the chosen arm can coincide with a reported one

    results = {"split": "heldout", "n": len(held_out), "tuned_on": f"{dev['dev_queries']} dev queries (seed {dev['seed']}); held-out run once, no tuning on it", "arms": {}}
    hits_by_arm = {}
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)

    for name, cfg, prior in arms:
        idx = run(held_out, cfg, prior)
        m, hits = metrics(idx, held_out)
        hits_by_arm[name] = hits
        results["arms"][name] = {**m, "cfg": cfg, "prior": prior}
        print(name, m, flush=True)
        np.save(out_dir / (re.sub(r"\W+", "_", name)[:60] + ".npy"), ticket_ids[idx])      # top-100 ticket ids per query, so scores can be recomputed

    baseline = hits_by_arm["bm25 clean (baseline)"]
    for name in hits_by_arm:
        if name != "bm25 clean (baseline)":
            results["arms"][name]["paired_vs_baseline"] = paired(hits_by_arm[name], baseline)

    json.dump(results, open(HERE / "results.json", "w"), indent=1, default=list)


if MODE == "tune":
    tune()
if MODE == "final":
    final()
