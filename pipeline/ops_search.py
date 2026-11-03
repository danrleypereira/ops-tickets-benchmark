"""Live demo: search the 320,296 serverfault tickets with a free-text question, then re-rank with a popularity signal.

What you see, in about 10 seconds:
  1. the 5 closest tickets by meaning (e5 embedding of the question + pgvector HNSW index), with their similarity score;
  2. the same search with the votes prior: tickets that already had many net votes move up (alpha 0.2, chosen on the dev queries).
The popularity signal is why Recall@10 went from 0.31 to 0.38 for this model on the held-out set.

Read-only: it only embeds your question on the CPU and runs SELECTs. Needs the container `ops-hnsw` running (docker start ops-hnsw).

usage: python -I pipeline/ops_search.py "your question"
"""
from pathlib import Path
import math
import pickle
import subprocess
import sys
import time

import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

question = ' '.join(sys.argv[1:]) or 'I deleted a big log file but df still says the disk is full'

import os
SF = Path(os.environ.get('OPS_BENCH_DIR', Path.home() / 'ops-benchmark')) / 'sf'
TICKETS = str(SF / 'gold/tickets.parquet')
VOTES = str(SF / 'votes_asof.pkl')    # {ticket id: (vote days, running net votes)} built from Votes.xml
TODAY = '2024-04-02'                                  # the day after the last day in the dump: "now" for this demo
ALPHA = 0.2                                           # weight of the votes bonus, chosen on dev queries (tuned on the dev queries)
CANDIDATES = 100                                      # how many hits the re-ranking looks at

started = time.time()


def lap(label):
    print(f'  [{time.time() - started:4.1f} s] {label}', flush=True)


# ----------------------------------------------------------------------------- embed the question (CPU)
model = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
model.max_seq_length = 256
vector = model.encode(['query: ' + question], normalize_embeddings=True, convert_to_numpy=True)[0]
lap('question embedded')

# ----------------------------------------------------------------------------- nearest neighbours in pgvector (HNSW)
vector_text = '[' + ','.join(f'{x:.6f}' for x in vector) + ']'
sql = (f"SET hnsw.ef_search=100; SELECT id, 1 - (v <=> '{vector_text}') FROM t ORDER BY v <=> '{vector_text}' LIMIT {CANDIDATES};")
done = subprocess.run(['docker', 'exec', '-i', 'ops-hnsw', 'psql', '-U', 'postgres', '-d', 'opsfhnsw', '-X', '-At', '-F', '|'],
                      input=sql, capture_output=True, text=True)
if done.returncode:
    sys.exit('could not reach the pgvector container (docker start ops-hnsw): ' + done.stderr[:200])
hits = [(int(line.split('|')[0]), float(line.split('|')[1])) for line in done.stdout.splitlines() if '|' in line]
lap(f'{len(hits)} nearest tickets found')

# ----------------------------------------------------------------------------- titles and votes
wanted = {i for i, _ in hits}
table = pq.read_table(TICKETS, columns=['id', 'title'])
title_of = {i: t for i, t in zip(table['id'].to_pylist(), table['title'].to_pylist()) if i in wanted}

with open(VOTES, 'rb') as fh:
    vote_history = pickle.load(fh)


def net_votes(ticket_id):
    """Net votes cast before TODAY (never below 0)."""
    days, running = vote_history.get(ticket_id, ([], []))
    placed = sum(1 for d in days if d < TODAY)
    return max(running[placed - 1], 0) if placed else 0


lap('titles and votes read')


def show(title, ranked):
    print(f'\n{title}')
    for place, (ticket_id, score, votes) in enumerate(ranked[:5], 1):
        print(f'  {place}. [{score:.3f}] ({votes:>3} votes) {title_of[ticket_id][:95]}')


# ----------------------------------------------------------------------------- plain ranking, then with the votes prior
print(f'\nQuestion: {question}')
plain = [(i, s, net_votes(i)) for i, s in hits]
show('1) By meaning only (similarity score)', plain)

best, worst = hits[0][1], hits[-1][1]
rescaled = lambda s: (s - worst) / max(best - worst, 1e-9)          # 0..1 inside these 100 hits
with_votes = sorted(plain, key=lambda row: -(rescaled(row[1]) + ALPHA * math.log1p(row[2])))
show('2) With the votes prior (similarity + popularity)', [(i, rescaled(s) + ALPHA * math.log1p(v), v) for i, s, v in with_votes])
lap('done')
