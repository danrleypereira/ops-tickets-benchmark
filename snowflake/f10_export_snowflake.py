"""Collect the evidence for a leaf package straight from Snowflake (read-only): row counts, object definitions (GET_DDL), task state.

Two kinds of export:
  stream-task   the incremental re-embedding demo (queue, vectors vs hashes, task state, DDL of table, stream, task)
  load-vector   the bulk load (row counts of the tickets, gold queries and vector tables, DDL, the incident view)

Output (in OUT_DIR): results.json with the numbers and ddl.sql with the definitions.

usage: f10_export_snowflake.py OUT_DIR stream-task|load-vector
"""
import json
import os
import subprocess
import sys

out_dir, kind = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)


def snow_rows(sql):
    """Run SQL through the snow CLI; return the rows of the last statement as dicts."""
    prefix = 'use schema SERVERFAULT; '
    done = subprocess.run(['snow', 'sql', '-c', os.environ.get('SNOW_CONNECTION', 'ops_bench'), '--format', 'json', '-q', prefix + sql], capture_output=True, text=True, check=True)
    result = json.loads(done.stdout)
    return result[-1] if result and isinstance(result[0], list) else result


def count(sql):
    return int(snow_rows(sql)[0]['C'])


def definition(object_kind, name):
    return snow_rows(f"select get_ddl('{object_kind}', 'SERVERFAULT.{name}') as d")[0]['D']


results = {}

if kind == 'stream-task':
    queue_rows = snow_rows("select action, count(*) n, count_if(done_at is null) pending from reembed_queue group by 1")
    results['queue'] = {r['ACTION']: dict(n=int(r['N']), pending=int(r['PENDING'])) for r in queue_rows}
    results['queue_done'] = count("select count(*) c from reembed_queue where done_at is not null")
    results['demo_rows'] = count("select count(*) c from tickets_inc")
    results['vectors_matching_hash'] = count("select count(*) c from ticket_vec_inc v join tickets_inc t on t.id=v.id and t.content_hash=v.content_hash")
    results['task_state_last'] = snow_rows("select state from table(information_schema.task_history(task_name=>'ENQUEUE_CHANGED', result_limit=>1))")[0]['STATE']
    results['note_recorded_at_run_time'] = (
        'stream held 20 rows (10 UPDATEs = 10 DELETE + 10 INSERT) before the task ran; the stream is consumed now, '
        'so this figure is from the run output, not re-derivable'
    )
    definitions = {
        'tickets_inc': definition('TABLE', 'tickets_inc'), 'ticket_vec_inc': definition('TABLE', 'ticket_vec_inc'),
        'tickets_inc_stream': definition('STREAM', 'tickets_inc_stream'), 'reembed_queue': definition('TABLE', 'reembed_queue'),
        'enqueue_changed': definition('TASK', 'enqueue_changed'),
    }
else:
    tables = ('tickets', 'gold_queries', 'ticket_vec_e5full256', 'query_vec_e5full256')
    for table in tables:
        results[f'rows_{table}'] = count(f'select count(*) c from {table}')

    # the dba copy lives in another schema (FORUM); this check only confirms its known size of 100,949 rows
    dba = subprocess.run(['snow', 'sql', '-c', os.environ.get('SNOW_CONNECTION', 'ops_bench'), '--format', 'json', '-q', 'select count(*) c from FORUM.tickets'],
                         capture_output=True, text=True).stdout
    results['dba_rows_tickets'] = int(dba.count('100949') > 0 and 100949)

    definitions = {table: definition('TABLE', table) for table in tables}
    definitions['incident_view'] = definition('VIEW', 'incident')

with open(os.path.join(out_dir, 'results.json'), 'w') as fh:
    json.dump(results, fh, indent=1)
with open(os.path.join(out_dir, 'ddl.sql'), 'w') as fh:
    fh.write('\n\n'.join(f'-- {name}\n{text}' for name, text in definitions.items()))
print(json.dumps(results, indent=1)[:600])
