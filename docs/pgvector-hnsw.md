# pgvector HNSW: the serving path (adds `f11_pgvector.py`, `docker-compose.hnsw.yml`, `ops_search.py`)
**What it is.** An approximate nearest-neighbour index (HNSW, m 16, ef_construction 64, cosine) over the 320,296 e5 vectors in an isolated Postgres 16 + pgvector 0.8 container, compared with exact search on the same vectors.

**Result** (1,079 held-out clean queries, one client, localhost, warm cache):

| ef_search | Recall@10 vs gold, out of 100 | Share of the exact top ten returned (average) | Latency p50 / p95 |
|---|---|---|---|
| 20 | 30.2 | 0.907 | 1.05 / 3.04 ms |
| 40 | 30.6 | 0.949 | 1.12 / 1.65 ms |
| 100 | 31.1 | 0.971 | 2.2 / 3.1 ms |
| 200 | 31.1 | 0.976 | 3.96 / 5.7 ms |
| 400 | 31.1 | 0.978 | 6.94 / 9.9 ms |

Exact search gives 30.9. At `ef_search` 100 the index returns on average 9.7 of the exact top ten (0.971), at about 2 ms per query; the index built in 70 s with 2 workers and takes 1,251 MB (table 19 MB). The 0.2-point difference above exact is noise, not a gain.

**How to run.** `docker compose -f pipeline/docker-compose.hnsw.yml up -d` then `python -I pipeline/f11_pgvector.py $SF /tmp/hnsw.json`. `ops_search.py "your question"` is a live search: it embeds the question on the CPU, searches the index, then re-ranks with the votes prior (needs the votes history: build it once with `python -I pipeline/votes_asof.py`, which needs `lxml` and `Votes.xml`); about 4.5 s end to end.

**Limits.** Single client; the first load time was not recorded; the container keeps its data under `OPS_PGDATA`.
