"""One-off repair: rewrite the tags column of tickets.parquet from posts.parquet.

Why it exists: the first run of f1_gold.py read tags with a pattern for the old "<a><b>" format, but these dumps write
"|a|b|", so every ticket came out with empty tags. f1_gold.py is fixed now; this script repaired the files that had
already been built, without redoing the whole run. The data-quality rule "no_tags_pct < 5" in f5_dq.py is what caught it.

usage: fix_tags.py PARQUET_DIR GOLD_DIR     (PARQUET_DIR has posts.parquet; GOLD_DIR has tickets.parquet, which is overwritten)
"""
import sys

import pyarrow as pa
import pyarrow.parquet as pq

parquet_dir, gold_dir = sys.argv[1:3]

# tags per question, as a space-separated string (the format f1_gold.py writes)
tags_of = {}
for batch in pq.ParquetFile(parquet_dir + '/posts.parquet').iter_batches(batch_size=50000, columns=['id', 'type', 'tags']):
    data = batch.to_pydict()
    for post_id, post_type, raw in zip(data['id'], data['type'], data['tags']):
        is_question = post_type == 1
        if is_question and raw:
            tags_of[post_id] = ' '.join(t for t in raw.replace('><', '|').strip('<>').split('|') if t)

tickets = pq.read_table(gold_dir + '/tickets.parquet')
ticket_ids = tickets['id'].to_pylist()

fixed_tags = pa.array([tags_of.get(i, '') for i in ticket_ids], pa.string())
tickets = tickets.set_column(tickets.schema.get_field_index('tags'), 'tags', fixed_tags)
pq.write_table(tickets, gold_dir + '/tickets.parquet', compression='zstd')

print('tickets', len(ticket_ids), 'with tags', sum(1 for i in ticket_ids if tags_of.get(i)))
