# Snowflake variant (needs a paid account) - DO NOT MERGE
Kept open as documentation; never merged. Everything here needs your own Snowflake account, which costs credits.

**What it is.** Load the tickets and the vectors into Snowflake (`VECTOR(FLOAT,768)` columns), exact search in SQL, the cost of that search, a stream + task that queues only changed tickets for re-embedding, and a hash-based sync into pgvector.

**Result** (320,296 tickets, 768-number vectors, result cache off; credits derived from elapsed time x warehouse rate, X-Small 1 and Large 8 credits per hour, not read from billing):

| Setup | Time per query | Credits per 1,000 queries |
|---|---|---|
| X-Small, 5 queries in one statement | 11.6 s | about 3.2 |
| X-Small, one query per statement | 14.4 s | about 4.0 |
| X-Small, 100 queries per statement (e5 title + question, 512) | 15.1 s | 4.21 |
| X-Small, 100 queries per statement (e5 full ticket, 512) | 15.6 s | 4.34 |
| Large, 200 queries in one statement | 2.04 s | about 4.5 |

Exact SQL search suits evaluating a few hundred queries, not serving users (pgvector HNSW answers in about 2 ms). Incremental re-embedding: editing 10 of 1,017 demo rows produced 20 stream rows (an update is a delete plus an insert), the task queued exactly the 10 inserts, a local worker re-embedded them, and 1,017 of 1,017 vectors matched their content hash; the pgvector sync moved 1,017 rows first, 0 on replay, 10 after the edits. Deletions and failure handling are not covered.

**How to run.** See the header comment of each file in `snowflake/`. The scripts take the role, warehouse and database from the connection (`ops_bench`, or `SNOW_CONNECTION`); create a least-privilege role first. The PUT scripts take `-D bench_dir=$OPS_BENCH_DIR`.

**Why it is not merged.** It needs a paid account and is not part of the open-source core.
