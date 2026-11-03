"""Turn the gold queries (two jsonl files) into one Parquet file that Snowflake can load.

Why: Snowflake's COPY INTO reads Parquet easily, and the list of original ids is stored as JSON text
(for example "[123, 456]") so it fits a plain string column; the SQL later opens it with PARSE_JSON.

Input : gold_dup_dev.jsonl and gold_dup_heldout.jsonl in GOLD_DIR (made by f1_gold.py).
Output: gold_queries.parquet in GOLD_DIR, used by f2_load.sql / f2_load_sf.sql.

usage: f2_gold_to_parquet.py GOLD_DIR
"""
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

gold_dir = sys.argv[1]

rows = []
for part in ('dev', 'heldout'):
    with open(os.path.join(gold_dir, f'gold_dup_{part}.jsonl')) as fh:
        for line in fh:
            query = json.loads(line)
            query['original_ids'] = json.dumps(query.pop('originals'))
            rows.append(query)

pq.write_table(pa.Table.from_pylist(rows), os.path.join(gold_dir, 'gold_queries.parquet'), compression='zstd')
print('gold queries written:', len(rows))
