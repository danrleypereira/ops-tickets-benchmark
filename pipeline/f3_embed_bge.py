"""bge-m3 vectors for the serverfault tickets and gold queries (a long GPU job, so it saves its progress).

Same text and 256-token window as the e5 run in f3_embed.py, so the two models are compared on equal terms.
bge-m3 needs no prefix. Vectors are fp16 and normalised (a dot product is the cosine similarity).

Checkpointing: tickets are embedded in length-sorted order in shards of SHARD tickets. Each finished shard is written to
OUT_DIR/shards/NNN.npy; if the job is stopped, running it again skips the shards that exist. When all shards are done
they are put back in file order and written in the same layout as f3_embed.py (columns id, vec), so f4_eval.py can read them.

Input : tickets.parquet and gold_dup_*.jsonl in GOLD_DIR.
Output: tickets.parquet, q_clean.parquet, q_raw.parquet, meta.json and shards/ in OUT_DIR.

usage: f3_embed_bge.py GOLD_DIR OUT_DIR [MAX_LEN=256] [VRAM_FRACTION=0.35] [BATCH=32]
run with the shared GPU environment:  python -I pipeline/f3_embed_bge.py ...
"""
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

gold, out = sys.argv[1:3]
max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 256
vram_fraction = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 32

SHARD = 20000
os.makedirs(os.path.join(out, 'shards'), exist_ok=True)
torch.cuda.set_per_process_memory_fraction(vram_fraction)   # the GPU is shared with other jobs

model = SentenceTransformer('BAAI/bge-m3', device='cuda')
model.max_seq_length = max_len
model.half()


def encode(texts):
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float16)   # fp16 on disk keeps the shards small; they are widened to float32 when the final file is written


def save(name, ids, vectors):
    wide = vectors.astype(np.float32)
    vec_column = pa.FixedSizeListArray.from_arrays(pa.array(wide.ravel(), pa.float32()), wide.shape[1])
    pq.write_table(pa.table({'id': pa.array(ids, pa.int64()), 'vec': vec_column}), os.path.join(out, name + '.parquet'), compression='zstd')


tickets = pq.read_table(os.path.join(gold, 'tickets.parquet'), columns=['id', 'ticket_text']).to_pydict()
ids, texts = tickets['id'], tickets['ticket_text']

# length order keeps each batch free of padding waste; kind='stable' makes the order, and so the shards, reproducible
order = np.argsort([len(t) for t in texts], kind='stable')
shard_count = (len(order) + SHARD - 1) // SHARD
started = time.time()
shards_resumed = 0

# ----------------------------------------------------------------------------- embed shard by shard
for shard in range(shard_count):
    shard_file = os.path.join(out, 'shards', f'{shard:03d}.npy')
    if os.path.exists(shard_file):
        shards_resumed += 1
        continue

    members = order[shard * SHARD:(shard + 1) * SHARD]
    shard_started = time.time()
    vectors = encode([texts[i] for i in members])

    np.save(shard_file + '.tmp.npy', vectors)
    os.replace(shard_file + '.tmp.npy', shard_file)   # rename last, so a half-written shard is never mistaken for a finished one

    seconds = time.time() - shard_started
    print(f'shard {shard + 1}/{shard_count} {len(members)} tickets {seconds:.0f}s ({len(members) / seconds:.1f}/s) '
          f'peak_alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)

# ----------------------------------------------------------------------------- put the shards back in file order
all_vectors = np.empty((len(order), 1024), np.float16)
for shard in range(shard_count):
    members = order[shard * SHARD:(shard + 1) * SHARD]
    all_vectors[members] = np.load(os.path.join(out, 'shards', f'{shard:03d}.npy'))
save('tickets', ids, all_vectors)
del all_vectors

# ----------------------------------------------------------------------------- gold queries (clean and raw)
queries = []
for part in ('dev', 'heldout'):
    with open(os.path.join(gold, f'gold_dup_{part}.jsonl')) as fh:
        queries += [json.loads(line) for line in fh]
query_ids = [q['qid'] for q in queries]

save('q_clean', query_ids, encode([q['query_clean'] for q in queries]))
save('q_raw', query_ids, encode([q['query_raw'] for q in queries]))

meta = dict(
    model='BAAI/bge-m3', prefix='', max_len=max_len, dims=1024, tickets=len(order), batch=batch_size, vram_fraction=vram_fraction,
    shards_resumed=shards_resumed, seconds_this_run=round(time.time() - started, 1),
    peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
)
with open(os.path.join(out, 'meta.json'), 'w') as fh:
    json.dump(meta, fh)
print(json.dumps(meta))
