# Many customers, one table: row-level security with pgvector

Two real tenants (Server Fault and Database Administrators questions) live in one table. The application connects as a role that **cannot** see the other tenant's rows, even if the query forgets its `WHERE`.

- `docker-compose.yml`: a throwaway Postgres 16 + pgvector on `127.0.0.1` (trust auth, no password, nothing but demo data).
- `sample_embed.py` and `load.py`: 20,000 random questions per site, embedded with all-MiniLM-L6-v2, loaded with HNSW, full-text and tenant indexes.
- `apply_rls.py`: two `NOSUPERUSER NOBYPASSRLS` roles, `ENABLE` + `FORCE ROW LEVEL SECURITY`, one policy that compares `tenant_id` with a setting bound per transaction.
- `leak_test.py`: 14 cases (own rows only, unbound/empty/blank tenant, a caller asking for the other tenant, vectors and titles taken from the other tenant, connection reuse, a cross-tenant write). `--positive-control` turns RLS off and checks that the test then fails.
- `real_overlap.py`: how much of a question's ten nearest neighbours belongs to the other site when nothing filters, so the test is not vacuous.

## Run
```bash
docker compose up -d                                           # PG_PORT=25433 by default
export PG_PORT=25433 OUT_DIR=./data
SERVERFAULT_POSTS=... DBA_POSTS=... python sample_embed.py     # skip if you already have sample.parquet and emb.npy
python load.py && python apply_rls.py
python leak_test.py                                            # expect 14 of 14, 0 leaked rows
python leak_test.py --positive-control                         # expect failures (RLS is switched back on at the end)
python real_overlap.py
```
Python 3.10+, `psycopg[binary]`, `pgvector`, `pyarrow`, `numpy`, `sentence-transformers`.

## What a run showed (one run, one host, 40,000 rows, pgvector 0.8)
- 14 of 14 cases pass, 0 leaked rows.
- Positive control (RLS off): 12 of 14 cases fail and 40,226 rows leak.
- With no filter, 7.25% of a dba question's ten nearest neighbours are Server Fault posts and 10.4% the other way round; the RLS reader returned 0 foreign rows for the same queries. These shares move a little between index builds (two earlier runs gave 6.4% and 10.45%, and 9.95%) because parallel HNSW builds are not deterministic. Leaks were 0 both times.

## Findings, not passes
- **The application can set its own tenant.** RLS protects against application bugs, not against a malicious application: bind the tenant in a trusted layer.
- **Short result lists.** An approximate index with a filter can return fewer than ten rows; that is lost recall, not a leak.
- **No connection pooler tested.** PgBouncer in transaction mode needs the same per-transaction binding (`set_config(..., true)`).

Data: Stack Exchange data dump, CC BY-SA 4.0 (Server Fault and Database Administrators). Not included in this repository.
