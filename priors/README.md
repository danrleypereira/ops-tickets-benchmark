# Votes prior (as-of) for duplicate-question search

Does the vote count of a candidate answer help rank it, **without using votes the system could not have known**?

- `votes_asof.py`: net votes per post counted only before the query's day, rebuilt from `Votes.xml`. The `score` column of the 2024 dump also counts votes cast after a duplicate was closed onto the original, which is future information.
- `r2.py`: sparse BM25 (Lucene idf, k1 1.5, b 0.75), title weighting and the prior; `python r2.py tune` picks settings on a dev slice, `python r2.py final` runs the held-out set once.
- `r2b_dense_prior.py`: the same prior on top of dense e5 vectors (min-max inside the top 100, then `alpha * log(1 + net votes)`).
- `eval_utils.py`: Recall@k, MRR and the paired bootstrap interval.

## Run
```bash
export DATA_DIR=/path/to/data      # gold/, parquet/posts.parquet, xml/Votes.xml, and the e5 embedding folders
python votes_asof.py               # not needed on its own: r2.py and r2b_dense_prior.py build and cache the table in $DATA_DIR/cache
python r2.py tune && python r2.py final
python r2b_dense_prior.py          # -> results_dense_prior.json (held-out run once, alpha chosen on dev)
```
Python 3.10+, numpy, scipy, scikit-learn, pyarrow, lxml. CPU only; `r2b_dense_prior.py` used 4.3 GB of RAM when I sampled it.

Data: Stack Exchange data dump (serverfault, dba), CC BY-SA 4.0.
