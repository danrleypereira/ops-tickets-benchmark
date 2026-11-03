"""F3: turn every ticket and every gold query into a vector on the GPU (local embeddings).

Input : tickets.parquet and gold_dup_{dev,heldout}.jsonl in GOLD_DIR (made by f1_gold.py).
Output (in OUT_DIR, float32 Parquet with columns id, vec):
  tickets.parquet   one vector per ticket
  q_clean.parquet   one vector per gold query, banner and links removed
  q_raw.parquet     one vector per gold query, text as found in the dump (to measure leakage)
  meta.json         model, window, speed, peak GPU memory

usage: f3_embed.py GOLD_DIR OUT_DIR MODEL PREFIX [MAX_LEN=256] [VRAM_FRACTION=0.3]
environment: F3_FIELD=ticket_text (default) or title_question; F3_BS=batch size (default 64)
run with the shared GPU environment:  python -I pipeline/f3_embed.py ...
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

gold, out, model_name, prefix = sys.argv[1:5]
max_len = int(sys.argv[5]) if len(sys.argv) > 5 else 256
vram_fraction = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
field = os.environ.get('F3_FIELD', 'ticket_text')
batch_size = int(os.environ.get('F3_BS', '64'))

# The GPU is shared with other jobs and a full GPU once crashed the desktop, so this process may use only a fraction.
torch.cuda.set_per_process_memory_fraction(vram_fraction)
os.makedirs(out, exist_ok=True)

model = SentenceTransformer(model_name, device='cuda')
model.max_seq_length = max_len  # longer text is cut at this many tokens
model.half()                    # fp16: about twice as fast, half the memory, no visible change in ranking


def encode(texts):
    """Vectors of unit length, so a dot product is the cosine similarity. e5 models need a 'query: ' prefix on both sides."""
    prefixed = [prefix + t for t in texts]
    vectors = model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float32)


def save(name, ids, vectors):
    vec_column = pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel(), pa.float32()), vectors.shape[1])
    table = pa.table({'id': pa.array(ids, pa.int64()), 'vec': vec_column})
    pq.write_table(table, os.path.join(out, name + '.parquet'), compression='zstd')


# ----------------------------------------------------------------------------- tickets
if field == 'title_question':
    # duplicates are question-to-question, so this variant embeds only the question; reading two small columns also saves RAM
    tickets = pq.read_table(os.path.join(gold, 'tickets.parquet'), columns=['id', 'title', 'question']).to_pydict()
    tickets['ticket_text'] = [title + '\n' + question for title, question in zip(tickets['title'], tickets['question'])]
else:
    tickets = pq.read_table(os.path.join(gold, 'tickets.parquet'), columns=['id', 'ticket_text']).to_pydict()

texts = tickets['ticket_text']
started = time.time()

# Sort by length so each batch holds texts of similar size and wastes little padding; put the vectors back in file order afterwards.
by_length = np.argsort([len(t) for t in texts])
sorted_vectors = encode([texts[i] for i in by_length])

position_in_sorted = np.empty(len(by_length), int)
position_in_sorted[by_length] = np.arange(len(by_length))
ticket_vectors = sorted_vectors[position_in_sorted]

save('tickets', tickets['id'], ticket_vectors)
ticket_seconds = time.time() - started

# ----------------------------------------------------------------------------- gold queries (clean and raw versions)
queries = []
for split in ('dev', 'heldout'):
    with open(os.path.join(gold, f'gold_dup_{split}.jsonl')) as fh:
        queries += [json.loads(line) for line in fh]
query_ids = [q['qid'] for q in queries]

save('q_clean', query_ids, encode([q['query_clean'] for q in queries]))
save('q_raw', query_ids, encode([q['query_raw'] for q in queries]))

# ----------------------------------------------------------------------------- run record
meta = dict(
    field=field, model=model_name, prefix=prefix, max_len=max_len, dims=int(ticket_vectors.shape[1]), tickets=len(texts),
    seconds_tickets=round(ticket_seconds, 1), tickets_per_s=round(len(texts) / ticket_seconds, 1),
    vram_fraction=vram_fraction, peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
)
with open(os.path.join(out, 'meta.json'), 'w') as fh:
    json.dump(meta, fh)
print(json.dumps(meta))
