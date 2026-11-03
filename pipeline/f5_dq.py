"""Data-quality checks to run on every load of the ticket table, with alert rules.

Checks: row count, duplicate ids, empty titles and questions, tickets without tags, bad dates, missing columns.
A rule that fails puts its name in "alerts" and sets "status" to ALERT. The tag rule exists because the first parse
once left every ticket without tags and nothing noticed.

Input : tickets.parquet (made by f1_gold.py).
Output: one JSON file with the measurements, the rules that were checked and the alerts.

usage: f5_dq.py TICKETS_PARQUET OUT_JSON [EXPECTED_ROWS]
"""
import json
import sys

import pyarrow.parquet as pq

path, out_path = sys.argv[1], sys.argv[2]
expected_rows = int(sys.argv[3]) if len(sys.argv) > 3 else None

EXPECTED_COLUMNS = ['id', 'title', 'tags', 'created', 'has_accepted', 'question', 'answer', 'comments', 'ticket_text']
FIRST_YEAR, LAST_YEAR = 2005, 2025   # Stack Exchange sites started in 2008; anything outside is a parsing error

parquet_file = pq.ParquetFile(path)
columns = parquet_file.schema_arrow.names


def is_blank(value):
    return not (value and value.strip())


def has_good_date(created):
    """created looks like '2012-03-31T10:00:00.000' and the year is plausible."""
    return bool(created and len(created) >= 10 and created[:4].isdigit() and FIRST_YEAR <= int(created[:4]) <= LAST_YEAR)


# ----------------------------------------------------------------------------- measure
# Read in batches and only the columns needed, so a 400 MB file is checked with little memory.
rows = duplicate_ids = empty_title = empty_question = no_tags = bad_date = with_accepted = 0
seen_ids = set()

for batch in parquet_file.iter_batches(batch_size=20000, columns=['id', 'title', 'tags', 'created', 'has_accepted', 'question']):
    data = batch.to_pydict()
    for ticket_id, title, tags, created, has_accepted, question in zip(
            data['id'], data['title'], data['tags'], data['created'], data['has_accepted'], data['question']):
        rows += 1

        if ticket_id in seen_ids:
            duplicate_ids += 1
        seen_ids.add(ticket_id)

        with_accepted += bool(has_accepted)
        empty_title += is_blank(title)
        empty_question += is_blank(question)
        no_tags += is_blank(tags)
        bad_date += not has_good_date(created)


def percent(count, digits):
    return round(100 * count / rows, digits)


measures = dict(
    file=path, rows=rows, unique_ids=len(seen_ids), duplicate_ids=duplicate_ids,
    empty_title_pct=percent(empty_title, 3), empty_question_pct=percent(empty_question, 3), no_tags_pct=percent(no_tags, 3),
    bad_date_pct=percent(bad_date, 3), with_accepted_pct=percent(with_accepted, 2),
    schema_missing=[c for c in EXPECTED_COLUMNS if c not in columns], schema_extra=[c for c in columns if c not in EXPECTED_COLUMNS],
)

# ----------------------------------------------------------------------------- alert rules (name, passes?)
rules = [
    ('duplicate_ids == 0', measures['duplicate_ids'] == 0),
    ('empty_title_pct < 0.1', measures['empty_title_pct'] < 0.1),
    ('empty_question_pct < 0.5', measures['empty_question_pct'] < 0.5),
    ('bad_date_pct < 0.1', measures['bad_date_pct'] < 0.1),
    ('no_tags_pct < 5', measures['no_tags_pct'] < 5),
    ('schema drift (missing columns) == []', not measures['schema_missing']),
]
if expected_rows:
    rules.append((f'rows == expected {expected_rows}', rows == expected_rows))

measures['alerts'] = [name for name, passes in rules if not passes]
measures['rules_checked'] = [name for name, _ in rules]
measures['status'] = 'OK' if not measures['alerts'] else 'ALERT'

with open(out_path, 'w') as fh:
    json.dump(measures, fh, indent=1)
print(json.dumps(measures))
