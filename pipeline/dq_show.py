"""Live demo: run the data-quality checks on the real ticket table and print each rule with its measured value and PASS / FAIL.

It calls f5_dq.py (read-only: it only reads tickets.parquet) and turns its JSON report into a table. The rule about tags exists
because the first parse once left every ticket without tags and nothing noticed.

usage: python -I pipeline/dq_show.py
"""
import sys
import json
import os
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TICKETS = os.path.join(os.environ.get('OPS_BENCH_DIR', os.path.expanduser('~/ops-benchmark')), 'sf/gold/tickets.parquet')
EXPECTED_ROWS = 320296
started = time.time()

report_path = os.path.join(tempfile.mkdtemp(), 'dq.json')
subprocess.run([sys.executable, '-I', os.path.join(HERE, 'f5_dq.py'), TICKETS, report_path, str(EXPECTED_ROWS)],
               check=True, capture_output=True)
with open(report_path) as fh:
    report = json.load(fh)

# the measured value behind each rule name
value_of = {
    'duplicate_ids': report['duplicate_ids'], 'empty_title_pct': report['empty_title_pct'], 'empty_question_pct': report['empty_question_pct'],
    'bad_date_pct': report['bad_date_pct'], 'no_tags_pct': report['no_tags_pct'], 'schema drift (missing columns)': report['schema_missing'] or 'none',
    'rows': report['rows'],
}

print(f"Data-quality checks on {report['rows']:,} tickets\n")
print(f"{'rule':40s} {'measured':>12s}   result")
for rule in report['rules_checked']:
    key = next(k for k in value_of if rule.startswith(k))
    result = 'FAIL' if rule in report['alerts'] else 'PASS'
    print(f'{rule:40s} {str(value_of[key]):>12s}   {result}')
print(f"\nStatus: {report['status']}   ({time.time() - started:.1f} s)")
