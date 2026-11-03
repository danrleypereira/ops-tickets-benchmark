"""Step 1: two real tenants for an isolation test.

Inputs : serverfault and dba questions as Parquet (columns id, type, title, text; type 1 = question), two paths below
Outputs: $OUT_DIR/sample.parquet  (tenant, post_id, title, body)
         $OUT_DIR/emb.npy         (40,000 x 384, normalised all-MiniLM-L6-v2 vectors)

20,000 random questions per site (seed 3). The embedding model does not matter for an isolation claim;
what matters is that the two sites really overlap in topics, so a missing filter would leak.
Data: Stack Exchange data dump, CC BY-SA 4.0 (Server Fault and Database Administrators). It is not in git.

  SERVERFAULT_POSTS=/path/sf/posts.parquet DBA_POSTS=/path/dba/posts.parquet OUT_DIR=./data python sample_embed.py
"""
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

OUT = Path(os.environ.get("OUT_DIR", "data"))
SITES = (("serverfault", Path(os.environ["SERVERFAULT_POSTS"])), ("dba", Path(os.environ["DBA_POSTS"])))
PER_SITE = 20000

OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(3)
rows = []          # (site, post_id, title, text)

for site, path in SITES:
    posts = pq.read_table(path, columns=["id", "type", "title", "text"]).to_pydict()

    # type 1 = question; very short texts are stubs, skip them
    candidates = [i for i, kind in enumerate(posts["type"]) if kind == 1 and posts["text"][i] and len(posts["text"][i]) > 40]

    for i in rng.choice(candidates, PER_SITE, replace=False):
        rows.append((site, int(posts["id"][i]), posts["title"][i] or "", posts["text"][i]))

sample = pa.table({
    "tenant": [r[0] for r in rows],
    "post_id": [r[1] for r in rows],
    "title": [r[2] for r in rows],
    "body": [r[3][:4000] for r in rows],       # stored body is capped; the embedding below uses the first 1,500 characters
})
pq.write_table(sample, OUT / "sample.parquet")

# 128 tokens is plenty for a title plus the opening of a question; CPU is enough for 40,000 texts
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
model.max_seq_length = 128
texts = [f"{title}\n{text[:1500]}" for _, _, title, text in rows]
vectors = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

np.save(OUT / "emb.npy", vectors)
print("rows", len(rows), vectors.shape)
