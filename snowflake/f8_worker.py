"""F8 local worker: re-embed the tickets that Snowflake queued as changed, and write the new vectors back.

The Snowflake task only builds the work queue (table reembed_queue). Embeddings are computed here, locally, with the same
model, prefix and 256-token window as the bulk run, so a re-embedded ticket is comparable with all the others.

It reads the queue through the snow CLI and does NOT change anything itself: it writes a SQL file (the vector updates and
the "done" marks) that you review and then run with `snow sql -f`.

usage: f8_worker.py OUT_SQL        (CPU only, RAM below 2 GB)
"""
import os
import json
import subprocess
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

out_sql = sys.argv[1]


def snow(sql):
    """Run SQL through the snow CLI and return the rows as dicts."""
    done = subprocess.run(['snow', 'sql', '-c', os.environ.get('SNOW_CONNECTION', 'ops_bench'), '--format', 'json', '-q', sql], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


# ----------------------------------------------------------------------------- the queued tickets
queued = snow(
    "use schema SERVERFAULT; "
    "select q.id, t.ticket_text, t.content_hash from reembed_queue q join tickets_inc t on t.id = q.id where q.done_at is null"
)
rows = queued[-1] if queued and isinstance(queued[0], list) else queued   # several statements come back as a list of result sets; the last one has the rows

# ----------------------------------------------------------------------------- embed
model = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
model.max_seq_length = 256
vectors = model.encode(['query: ' + r['TICKET_TEXT'] for r in rows], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

# ----------------------------------------------------------------------------- the SQL to run
statements = ['USE SCHEMA SERVERFAULT;']
for row, vector in zip(rows, vectors):
    numbers = ','.join(f'{x:.6f}' for x in vector)
    statements.append(
        f"UPDATE ticket_vec_inc SET vec = [{numbers}]::VECTOR(FLOAT,768), content_hash = '{row['CONTENT_HASH']}' WHERE id = {row['ID']};"
    )
    statements.append(f"UPDATE reembed_queue SET done_at = CURRENT_TIMESTAMP() WHERE id = {row['ID']} AND done_at IS NULL;")

# final check: every queue row is done and every stored vector matches the current text of its ticket
statements.append(
    "SELECT count(*) AS done FROM reembed_queue WHERE done_at IS NOT NULL; "
    "SELECT count(*) AS vectors_matching_hash FROM ticket_vec_inc v JOIN tickets_inc t ON t.id = v.id AND t.content_hash = v.content_hash;"
)

with open(out_sql, 'w') as fh:
    fh.write('\n'.join(statements))
print('queue rows embedded:', len(rows))
