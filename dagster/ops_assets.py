"""Dagster asset graph for the ops-tickets benchmark, from the raw dump (F0) to the Snowflake cost summary (F9).

The design choice that matters: each asset either RUNS its step (when the step is cheap) or VERIFIES the saved result of the
step (when it is heavy). Cheap steps: data-quality checks, Snowflake row counts, dbt run + test. Heavy steps: the download, the
embeddings, the evaluation, the HNSW index. Verifying means checking that the output exists and recording its size, hash and
headline numbers, so a click in the Dagster UI never re-spends GPU hours or Snowflake credits by accident. The command that
re-runs a heavy step is written in that asset's description.

Credentials: with_snowflake_env.py puts them in the environment of the process at run time; none are stored in this file.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dagster import Definitions, Output, asset

SF = Path(os.environ.get('OPS_BENCH_DIR', Path.home() / 'ops-benchmark')) / 'sf'
ROOT = Path(__file__).resolve().parent.parent   # the repository root
PYTHON = sys.executable
DBT = os.environ.get('DBT_BIN') or (str(Path(sys.executable).parent / 'dbt') if (Path(sys.executable).parent / 'dbt').exists() else shutil.which('dbt') or 'dbt')   # dbt next to this interpreter, else on PATH
MAX_HASHED_BYTES = 3e9   # hashing a multi-GB file would take minutes; larger files are only measured by size


def short_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()[:16]


def verify(path):
    """Check that a saved output exists and describe it (size and hash) for the Dagster metadata."""
    file = Path(path)
    assert file.exists(), f'missing {file}: re-run the step'

    small_enough = file.is_file() and file.stat().st_size < MAX_HASHED_BYTES
    return {'path': str(file), 'bytes': file.stat().st_size, 'sha256_16': short_sha256(file) if small_enough else 'skipped'}


def snowflake_rows(sql):
    """Run SQL in the benchmark schema through the snow CLI; return the rows of the last statement as dicts."""
    done = subprocess.run(['snow', 'sql', '-c', os.environ.get('SNOW_CONNECTION', 'ops_bench'), '--format', 'json', '-q', 'use schema SERVERFAULT; ' + sql],
                          capture_output=True, text=True, check=True)
    result = json.loads(done.stdout)
    return result[-1] if result and isinstance(result[0], list) else result


# ----------------------------------------------------------------------------- data (heavy: verified, not re-run)
@asset(description='F0 raw dump: serverfault.com.7z from archive.org (CC BY-SA 4.0). Re-run: curl -L -O https://archive.org/download/stackexchange/serverfault.com.7z')
def raw_dump():
    return Output(None, metadata=verify(SF / 'raw/serverfault.com.7z'))


@asset(deps=[raw_dump], description='F0 parquet tables (posts, postlinks, comments, tags). Re-run: python f0_parse.py xml parquet')
def parquet_tables():
    return Output(None, metadata=verify(SF / 'parquet/posts.parquet'))


@asset(deps=[parquet_tables], description='F1 ticket base + duplicate gold (dev/held-out, seed 13). Re-run: python f1_gold.py parquet gold')
def tickets_and_gold():
    with open(SF / 'gold/f1_stats.json') as fh:
        stats = json.load(fh)
    return Output(None, metadata={**verify(SF / 'gold/tickets.parquet'), **stats})


# ----------------------------------------------------------------------------- checks and loads (cheap: run every time)
@asset(deps=[tickets_and_gold], description='Per-load data-quality checks with alert rules (runs now). A failing rule fails the asset.')
def dq_report():
    report_path = SF / 'results/dq_serverfault_dagster.json'
    subprocess.run([PYTHON, '-I', str(ROOT / 'pipeline/f5_dq.py'), str(SF / 'gold/tickets.parquet'), str(report_path), '320296'], check=True, capture_output=True)

    with open(report_path) as fh:
        report = json.load(fh)
    assert report['status'] == 'OK', report['alerts']    # an alert stops the pipeline here

    return Output(None, metadata={key: report[key] for key in ('rows', 'duplicate_ids', 'no_tags_pct', 'status')})


@asset(deps=[dq_report], description='F2 Snowflake load: row counts of tickets and gold_queries (runs now). Re-run: snow sql -f f2_load_sf.sql')
def snowflake_tables():
    tickets = int(snowflake_rows('select count(*) c from tickets')[0]['C'])
    gold_queries = int(snowflake_rows('select count(*) c from gold_queries')[0]['C'])
    assert (tickets, gold_queries) == (320296, 4790), (tickets, gold_queries)
    return Output(None, metadata={'tickets': tickets, 'gold_queries': gold_queries})


# ----------------------------------------------------------------------------- vectors and retrieval (heavy: verified)
@asset(deps=[tickets_and_gold], description='F3 local embeddings, e5-base full ticket @256 (GPU). Re-run: python -I f3_embed.py gold emb_e5base intfloat/multilingual-e5-base "query: " 256 0.3')
def embeddings_e5():
    with open(SF / 'emb_e5base/meta.json') as fh:
        meta = json.load(fh)
    return Output(None, metadata={**verify(SF / 'emb_e5base/tickets.parquet'), **meta})


@asset(deps=[embeddings_e5, snowflake_tables], description='F3b vectors in Snowflake VECTOR(FLOAT,768) columns (runs now: counts). Re-run: snow sql -f f3_vec_load.sql')
def snowflake_vectors():
    vectors = int(snowflake_rows('select count(*) c from ticket_vec_e5full256')[0]['C'])
    assert vectors == 320296, vectors
    return Output(None, metadata={'ticket_vectors': vectors})


@asset(deps=[embeddings_e5], description='F4/F6 evaluation of BM25, dense and RRF on the held-out queries with paired bootstrap. Re-run: python -I f4_eval.py gold emb_e5base results/e5full256_heldout.json heldout')
def retrieval_eval():
    with open(SF / 'results/e5full256_heldout.json') as fh:
        results = json.load(fh)
    return Output(None, metadata={
        'clean_dense_R10': results['clean/dense']['R10'], 'clean_bm25_R10': results['clean/bm25']['R10'], 'clean_rrf_R10': results['clean/rrf']['R10'],
        'leakage_dense_R10_diff': results['paired/leakage/dense(raw-clean)']['R10']['diff'],
    })


@asset(deps=[embeddings_e5], description='pgvector HNSW arm (isolated container). Re-run: docker compose -f docker-compose.hnsw.yml up -d; python -I f11_pgvector.py ...')
def pgvector_hnsw():
    with open(SF / 'results/pgvector_hnsw.json') as fh:
        results = json.load(fh)
    at_ef100 = results['sweep']['100']
    return Output(None, metadata={'build_seconds': results['hnsw_build_seconds'], 'ef100_overlap': at_ef100['overlap_with_exact_top10'], 'ef100_p50_ms': at_ef100['latency_ms_p50']})


# ----------------------------------------------------------------------------- transformation and cost
@asset(deps=[snowflake_vectors, snowflake_tables], description='dbt models and tests on Snowflake (runs now): dbt run + dbt test')
def dbt_models():
    for command in ('run', 'test'):
        done = subprocess.run([sys.executable, str(ROOT / 'snowflake/with_snowflake_env.py'), DBT, command, '--profiles-dir', '.', '--project-dir', '.'],
                              cwd=ROOT / 'dbt', capture_output=True, text=True)
        assert done.returncode == 0, done.stdout[-800:]
    return Output(None, metadata={'dbt': 'run and test passed'})


@asset(deps=[snowflake_vectors], description='F9 cost of brute-force vector SQL (timings from the saved scale test, credits derived)')
def snowflake_cost():
    with open(SF / 'results/snowflake_export/load-vector/results.json') as fh:
        derived = json.load(fh)['derived']
    return Output(None, metadata=derived)


defs = Definitions(assets=[raw_dump, parquet_tables, tickets_and_gold, dq_report, snowflake_tables, embeddings_e5, snowflake_vectors,
                           retrieval_eval, pgvector_hnsw, dbt_models, snowflake_cost])
