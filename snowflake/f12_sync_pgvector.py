"""Keep pgvector in step with Snowflake by moving only the rows that changed (hash-based, safe to repeat).

Demo on the 1,017-ticket table, in five phases:
  1. first sync        every row is new, so every row moves
  2. replay            nothing changed, so nothing moves
  3. edit 10 rows      Snowflake's stream + task queue them, f8_worker.py re-embeds them locally and writes the vectors back
  4. sync again        exactly those 10 rows move
  5. replay            nothing moves

How it stays cheap: compare (id, content_hash) first, which is tiny; fetch vectors only for ids whose hash differs;
upsert with ON CONFLICT ... WHERE hash differs, so repeating a batch changes nothing (a retry after a failure is safe).
Not covered: deletions (a stream row with action DELETE would become a DELETE in pgvector) and failure handling.

Needs: container ops-hnsw running, Snowflake connection ops_bench (or $SNOW_CONNECTION), tables from f8_stream_task.sql. CPU only, small RAM.

usage: f12_sync_pgvector.py OUT_JSON
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
out_path = sys.argv[1]

BATCH = 50   # ids per Snowflake round trip
DOCKER_PSQL = ['docker', 'exec', '-i', 'ops-hnsw', 'psql', '-U', 'postgres', '-d', 'opsfhnsw', '-X']


def snow(sql):
    """Run SQL in Snowflake through the snow CLI; return the rows of the last statement as dicts."""
    prefix = 'use schema SERVERFAULT; '
    done = subprocess.run(['snow', 'sql', '-c', os.environ.get('SNOW_CONNECTION', 'ops_bench'), '--format', 'json', '-q', prefix + sql], capture_output=True, text=True, check=True)
    result = json.loads(done.stdout)
    return result[-1] if result and isinstance(result[0], list) else result


def pg(sql):
    done = subprocess.run(DOCKER_PSQL + ['-At', '-v', 'ON_ERROR_STOP=1', '-c', sql], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(done.stderr[:300])
    return done.stdout


pg('CREATE TABLE IF NOT EXISTS t_inc (id bigint PRIMARY KEY, h text NOT NULL, v vector(768) NOT NULL, synced_at timestamptz DEFAULT now())')


def sync(label):
    started = time.time()

    snowflake_hash = {int(r['ID']): r['CONTENT_HASH'] for r in snow('select id, content_hash from ticket_vec_inc')}
    pgvector_hash = {int(line.split('|')[0]): line.split('|')[1] for line in pg('select id, h from t_inc').splitlines() if line}

    to_move = [i for i, h in snowflake_hash.items() if pgvector_hash.get(i) != h]

    moved = 0
    for start in range(0, len(to_move), BATCH):
        batch_ids = to_move[start:start + BATCH]
        rows = snow(f"select id, content_hash, vec::ARRAY::VARCHAR as v from ticket_vec_inc where id in ({','.join(map(str, batch_ids))})")

        upserts = '\n'.join(
            f"INSERT INTO t_inc (id, h, v) VALUES ({r['ID']}, '{r['CONTENT_HASH']}', '{r['V'].replace(' ', '')}') "
            f"ON CONFLICT (id) DO UPDATE SET h = EXCLUDED.h, v = EXCLUDED.v, synced_at = now() WHERE t_inc.h IS DISTINCT FROM EXCLUDED.h;"
            for r in rows
        )
        subprocess.run(DOCKER_PSQL + ['-q', '-v', 'ON_ERROR_STOP=1'], input=upserts, text=True, check=True, capture_output=True)
        moved += len(rows)

    outcome = dict(label=label, snowflake_rows=len(snowflake_hash), pg_rows_before=len(pgvector_hash), hash_differs=len(to_move),
                   rows_moved=moved, seconds=round(time.time() - started, 1))
    print(outcome, flush=True)
    return outcome


# ----------------------------------------------------------------------------- phases 1 and 2
log = [sync('1 initial sync'), sync('2 replay, nothing changed')]

# ----------------------------------------------------------------------------- phase 3: edit 10 rows, let the stream + task queue them
snow("update tickets_inc set ticket_text = ticket_text || ' [edited again]', content_hash = sha2(ticket_text || ' [edited again]') "
     "where id in (select id from tickets_inc order by id limit 10 offset 10)")
snow("execute task enqueue_changed")
time.sleep(25)   # the task runs asynchronously; give it time to fill the queue
queued = snow("select count(*) c from reembed_queue where done_at is null")[0]['C']
print('queued for re-embedding:', queued)

# the local worker embeds the queued rows and writes a SQL file with the new vectors; run that file in Snowflake
writeback_sql = '/tmp/f12_writeback.sql'
subprocess.run([PYTHON, '-I', os.path.join(HERE, 'f8_worker.py'), writeback_sql], check=True,
               env=dict(os.environ, HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES=''), capture_output=True)
subprocess.run(['snow', 'sql', '-c', os.environ.get('SNOW_CONNECTION', 'ops_bench'), '-f', writeback_sql], check=True, capture_output=True)

# ----------------------------------------------------------------------------- phases 4 and 5
log += [sync('3 after 10 edits + local re-embed'), sync('4 replay')]

pg_rows = pg('select count(*) from t_inc')
snowflake_rows = snow('select count(*) c from ticket_vec_inc')[0]['C']
with open(out_path, 'w') as fh:
    json.dump(dict(steps=log, pg_rows=int(pg_rows), snowflake_rows=int(snowflake_rows), queue_rows_for_edits=int(queued)), fh, indent=1)
print('done', pg_rows.strip(), snowflake_rows)
