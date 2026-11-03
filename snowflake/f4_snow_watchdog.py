"""Time budget for the big Snowflake statement.

The batched brute-force top-k statement ran on a LARGE warehouse, which bills while it works. This watchdog polls it:
if the statement is still running after MAX_SECONDS it is cancelled, and in every case the warehouse is set back to
X-Small so nothing keeps billing at the larger size.

usage: f4_snow_watchdog.py MAX_SECONDS        (polls through the snow CLI every 30 s; prints one line per state change)
"""
import os
import json
import subprocess
import sys
import time

max_seconds = int(sys.argv[1])
POLL_SECONDS = 30


def query(sql):
    """Run SQL through the snow CLI; return the rows as dicts, or [] if anything goes wrong (a failed poll must not stop the watchdog)."""
    done = subprocess.run(['snow', 'sql', '-c', os.environ.get('SNOW_CONNECTION', 'ops_bench'), '--format', 'json', '-q', sql], capture_output=True, text=True)
    try:
        result = json.loads(done.stdout)
        return result[-1] if result and isinstance(result[0], list) else result
    except Exception:
        return []


latest_state = None
while True:
    rows = query(
        "select query_id, execution_status st, datediff('second', start_time, current_timestamp()) sec "
        "from table(information_schema.query_history(result_limit=>20)) "
        "where query_text ilike 'CREATE OR REPLACE TABLE snow_topk%' order by start_time desc limit 1"
    )

    if rows:
        state, seconds, query_id = rows[0]['ST'], int(rows[0]['SEC']), rows[0]['QUERY_ID']

        if state != latest_state:
            print(time.strftime('%H:%M:%S'), query_id, state, seconds, flush=True)
            latest_state = state

        if state != 'RUNNING':
            break

        if seconds >= max_seconds:
            print('budget exceeded, cancelling', query_id, flush=True)
            query(f"select system$cancel_query('{query_id}')")
            time.sleep(20)   # give Snowflake a moment to stop the statement before resizing
            break

    time.sleep(POLL_SECONDS)

query("set wh = current_warehouse(); alter warehouse identifier($wh) set warehouse_size = XSMALL auto_suspend = 60")
print('warehouse restored to XSMALL', flush=True)
