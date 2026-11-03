"""F7: two-hop questions answered with the help of related-ticket links (serverfault).

Each question has two tickets that must both be found, and the moderators linked them to each other ("related", PostLinks type 1).
Duplicate links (type 3) are never used as edges: they are the labels of the other benchmark.

Arms, all starting from the same dense (e5) search:
  dense  the plain top 10
  link1  take the top-5 hits as seeds, add every ticket linked to a seed, re-rank everything by its own similarity
  path   like link1, but a linked ticket also gets half the similarity of its best linked seed  (version 1)
  path2  like path, but a seed gets the same bonus using its own similarity                      (version 2)
Version 1 gave only neighbours a bonus, which pushed an exact single-hop answer out of the top 10 (found in the first run); version 2 fixes that.

Metric: path recall@5 = share of the 2 needed tickets found in the top 5 (1.0 = "full path"); also recall@10.

Input : SF_DIR with emb_e5base/tickets.parquet and parquet/postlinks.parquet; GOLD_JSONL with the questions.
Output: OUT_JSON with a summary per split and arm and the top-10 per question.
RAM about 1.5 GB, CPU only.

usage: f7_graph.py SF_DIR GOLD_JSONL OUT_JSON [SPLITS comma list]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

sf_dir, gold_path, out_path = sys.argv[1:4]
only_splits = sys.argv[4].split(',') if len(sys.argv) > 4 else None

BONUS_WEIGHT = 0.5   # fixed in advance; the held-out questions were run once, so it was never tuned
SEEDS = 5


# ----------------------------------------------------------------------------- ticket vectors
vector_file = pq.ParquetFile(os.path.join(sf_dir, 'emb_e5base/tickets.parquet'))
ticket_vectors = np.empty((vector_file.metadata.num_rows, 768), np.float32)
ticket_ids = []
filled = 0
for batch in vector_file.iter_batches(batch_size=20000):   # in batches: converting the whole column at once would double the memory
    data = batch.to_pydict()
    count = len(data['id'])
    ticket_ids += data['id']
    ticket_vectors[filled:filled + count] = np.array(data['vec'], np.float32)
    filled += count

ticket_ids = np.array(ticket_ids)
position_of = {int(i): j for j, i in enumerate(ticket_ids)}
ticket_vectors = torch.from_numpy(ticket_vectors)


# ----------------------------------------------------------------------------- the link graph (related links only)
linked = defaultdict(set)
for batch in pq.ParquetFile(os.path.join(sf_dir, 'parquet/postlinks.parquet')).iter_batches(columns=['post', 'related', 'type']):
    data = batch.to_pydict()
    for post, related, link_type in zip(data['post'], data['related'], data['type']):
        in_base = post in position_of and related in position_of
        if link_type == 1 and in_base:
            linked[post].add(related)
            linked[related].add(post)    # the link is read in both directions


# ----------------------------------------------------------------------------- questions
with open(gold_path) as fh:
    questions = [json.loads(line) for line in fh]
if only_splits:
    questions = [q for q in questions if q['split'] in only_splits]

encoder = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
encoder.max_seq_length = 256
question_vectors = encoder.encode(['query: ' + q['q'] for q in questions], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


# ----------------------------------------------------------------------------- the four arms
def run_arms(query_vector):
    similarity = (ticket_vectors @ torch.from_numpy(query_vector)).numpy()
    similar_to_query = lambda ticket: float(similarity[position_of[ticket]])

    best_first = np.argsort(-similarity)[:10]
    dense = [int(ticket_ids[i]) for i in best_first]

    seeds = dense[:SEEDS]
    candidates = set(dense)
    for seed in seeds:
        candidates |= {neighbour for neighbour in linked[seed]}

    link1 = sorted(candidates, key=lambda t: -similar_to_query(t))[:10]

    # version 1: a ticket's bonus is the best similarity among the seeds it is linked to (seeds themselves get none)
    bonus = {t: max([similar_to_query(s) for s in linked[t] if s in seeds] or [0.0]) for t in candidates}
    path = sorted(candidates, key=lambda t: -(similar_to_query(t) + BONUS_WEIGHT * bonus[t]))[:10]

    # version 2: a seed also gets a bonus, its own similarity, so exact answers are not pushed down by their neighbours
    bonus_with_seeds = {t: max(bonus[t], similar_to_query(t) if t in seeds else 0.0) for t in candidates}
    path2 = sorted(candidates, key=lambda t: -(similar_to_query(t) + BONUS_WEIGHT * bonus_with_seeds[t]))[:10]

    return dict(dense=dense, link1=link1, path=path, path2=path2)


per_question = []
for question, query_vector in zip(questions, question_vectors):
    arms = run_arms(query_vector)
    needed = set(question['relevant'])

    row = dict(qid=question['qid'], split=question['split'], type=question['type'])
    for arm, ranking in arms.items():
        row[arm] = dict(
            r5=len(needed & set(ranking[:5])) / len(needed),
            r10=len(needed & set(ranking[:10])) / len(needed),
            full5=int(needed <= set(ranking[:5])),
            rank=ranking[:10],
        )
    per_question.append(row)


# ----------------------------------------------------------------------------- summary per split and arm
summary = {}
for split in sorted({row['split'] for row in per_question}):
    rows = [row for row in per_question if row['split'] == split]
    for arm in ('dense', 'link1', 'path', 'path2'):
        summary[f'{split}/{arm}'] = dict(
            n=len(rows),
            path_recall5=round(float(np.mean([row[arm]['r5'] for row in rows])), 3),
            full_paths5=int(sum(row[arm]['full5'] for row in rows)),
            recall10=round(float(np.mean([row[arm]['r10'] for row in rows])), 3),
        )

with open(out_path, 'w') as fh:
    json.dump(dict(summary=summary, per_question=per_question), fh, indent=1)
for key, value in summary.items():
    print(key, value)
