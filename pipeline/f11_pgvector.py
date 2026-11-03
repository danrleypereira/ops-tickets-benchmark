"""pgvector HNSW arm: how good and how fast is an approximate index on the same vectors?

Steps:
  1. load the 320,296 e5 vectors into an isolated pgvector container (docker-compose.hnsw.yml, container ops-hnsw);
  2. build an HNSW index (m 16, ef_construction 64, cosine);
  3. for several ef_search settings, run the held-out queries and record
       - Recall@10 against the gold originals,
       - overlap with the EXACT top-10 (from f4_eval.py), which says how much the index loses against exact search,
       - latency per query (p50 and p95, one client on localhost, warm cache).

Input : SF_DIR with emb_e5base/tickets.parquet, gold/gold_dup_heldout.jsonl, emb_e5base/q_clean.parquet and
        results/e5full256_heldout_outputs/clean_dense.npy (the exact ranking).
Output: OUT_JSON with build time, index size and the ef_search sweep.
The load is streamed, so this process needs under 1.5 GB of RAM. If the table is already loaded it is reused.

usage: f11_pgvector.py SF_DIR OUT_JSON
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sf_dir, out_path = sys.argv[1:3]

EXPECTED_ROWS = 320296
DOCKER_PSQL = ['docker', 'exec', '-i', 'ops-hnsw', 'psql', '-U', 'postgres', '-d', 'opsfhnsw', '-X']


def psql(sql):
    """Run one SQL string in the container; raise on error, return the output without table decoration."""
    done = subprocess.run(DOCKER_PSQL + ['-At', '-v', 'ON_ERROR_STOP=1', '-c', sql], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(done.stderr[:400])
    return done.stdout


# ----------------------------------------------------------------------------- 1. load (skipped if the table is already full)
try:
    rows_in_table = int(psql('select count(*) from t').strip())
except RuntimeError:
    rows_in_table = 0

if rows_in_table == EXPECTED_ROWS:
    rows, load_seconds = rows_in_table, None   # loaded by an earlier run (its index build had failed on shared memory)
else:
    psql('DROP TABLE IF EXISTS t; CREATE TABLE t (id bigint PRIMARY KEY, v vector(768))')

    started = time.time()
    # COPY from stdin is far faster than row-by-row INSERT; one text line per ticket: id <tab> [v1,v2,...]
    copy = subprocess.Popen(DOCKER_PSQL + ['-v', 'ON_ERROR_STOP=1', '-c', 'COPY t FROM STDIN'], stdin=subprocess.PIPE, text=True)
    rows = 0
    for batch in pq.ParquetFile(os.path.join(sf_dir, 'emb_e5base/tickets.parquet')).iter_batches(batch_size=5000):
        data = batch.to_pydict()
        for ticket_id, vector in zip(data['id'], data['vec']):
            numbers = ','.join(f'{x:.6f}' for x in vector)
            copy.stdin.write(f'{ticket_id}\t[{numbers}]\n')
            rows += 1
    copy.stdin.close()
    assert copy.wait() == 0
    load_seconds = time.time() - started

# ----------------------------------------------------------------------------- 2. build the index
started = time.time()
psql("DROP INDEX IF EXISTS t_hnsw; SET maintenance_work_mem='2GB'; "
     "CREATE INDEX t_hnsw ON t USING hnsw (v vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
build_seconds = time.time() - started
sizes = psql("select pg_size_pretty(pg_relation_size('t_hnsw')), pg_size_pretty(pg_relation_size('t'))").strip()

# ----------------------------------------------------------------------------- 3. queries and the exact reference
with open(os.path.join(sf_dir, 'gold/gold_dup_heldout.jsonl')) as fh:
    queries = [json.loads(line) for line in fh]

query_table = pq.read_table(os.path.join(sf_dir, 'emb_e5base/q_clean.parquet'))
query_vectors = np.stack(query_table['vec'].to_numpy(zero_copy_only=False))
query_row = {i: j for j, i in enumerate(query_table['id'].to_pylist())}

# exact cosine top-10 for the same queries, in the same order (saved by f4_eval.py)
exact_top10 = np.load(os.path.join(sf_dir, 'results/e5full256_heldout_outputs/clean_dense.npy'))[:, :10]


def run_all_queries(ef_search, k=10):
    """Send every held-out query in one psql session with \\timing on; return the ids found and the milliseconds per query."""
    script = [f'SET hnsw.ef_search={ef_search}; SET max_parallel_workers_per_gather=0;', '\\timing on']
    for q in queries:
        vector = ','.join(f'{x:.6f}' for x in query_vectors[query_row[q['qid']]])
        script.append(f"SELECT id FROM t ORDER BY v <=> '[{vector}]' LIMIT {k};")

    output = subprocess.run(DOCKER_PSQL + ['-At'], input='\n'.join(script), capture_output=True, text=True).stdout.splitlines()

    found, milliseconds, current = [], [], []
    for line in output:
        if line.startswith('Time:'):                  # psql prints the time after each query's rows
            milliseconds.append(float(line.split()[1]))
            found.append(current)
            current = []
        elif line.strip().isdigit():
            current.append(int(line))
    return found[-len(queries):], milliseconds[-len(queries):]


results = dict(
    rows=rows,
    load_seconds=round(load_seconds, 1) if load_seconds is not None else 'not recorded (reused table from the first run)',
    hnsw_build_seconds=round(build_seconds, 1), sizes_index_table=sizes,
    params=dict(m=16, ef_construction=64, metric='cosine'), sweep={},
)

for ef_search in (20, 40, 100, 200, 400):   # higher = looks at more candidates: better recall, slower
    found, milliseconds = run_all_queries(ef_search)

    hits, overlaps = 0.0, []
    for ids, q, exact in zip(found, queries, exact_top10):
        hits += bool(set(q['originals']) & set(ids))
        overlaps.append(len(set(ids) & set(int(x) for x in exact)) / 10)

    results['sweep'][str(ef_search)] = dict(
        R10_vs_gold=round(hits / len(queries), 4), overlap_with_exact_top10=round(float(np.mean(overlaps)), 4),
        latency_ms_p50=round(float(np.percentile(milliseconds, 50)), 2), latency_ms_p95=round(float(np.percentile(milliseconds, 95)), 2),
        queries=len(milliseconds),
    )
    print(ef_search, results['sweep'][str(ef_search)], flush=True)

with open(out_path, 'w') as fh:
    json.dump(results, fh, indent=1)
print(json.dumps({k: v for k, v in results.items() if k != 'sweep'}))
