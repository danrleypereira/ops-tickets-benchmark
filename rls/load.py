"""Step 2: load the two real tenants into pgvector.

Table ops.tickets(id, tenant_id, post_id, title, body, embedding vector(384), tsv). Row-level security is applied afterwards by apply_rls.py.
  OUT_DIR=./data python load.py        (needs the container from docker-compose.yml)
"""
import os
import time
from pathlib import Path

import numpy as np
import psycopg
import pyarrow.parquet as pq
from pgvector.psycopg import register_vector

from common import DB, connect

OUT = Path(os.environ.get("OUT_DIR", "data"))

with connect(dbname="postgres", autocommit=True) as admin:
    if not admin.execute("select 1 from pg_database where datname=%s", (DB,)).fetchone():
        admin.execute(psycopg.sql.SQL("create database {}").format(psycopg.sql.Identifier(DB)))

sample = pq.read_table(OUT / "sample.parquet").to_pydict()
vectors = np.load(OUT / "emb.npy")

with connect(autocommit=True) as db:
    db.execute("create extension if not exists vector")
    register_vector(db)
    db.execute("create schema if not exists ops")
    db.execute("drop table if exists ops.tickets")

    # the tsvector column is generated, so full-text search needs no extra load step
    db.execute("""create table ops.tickets (id bigserial primary key, tenant_id text not null, post_id bigint not null, title text, body text, embedding vector(384),
                  tsv tsvector generated always as (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(body,''))) stored, unique (tenant_id, post_id))""")

    start = time.time()

    # binary COPY is far faster than row-by-row INSERT for 40,000 vectors
    with db.cursor().copy("COPY ops.tickets (tenant_id, post_id, title, body, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(["text", "int8", "text", "text", "vector"])
        for i in range(len(vectors)):
            body = sample["body"][i].replace("\x00", "")     # Postgres text cannot hold NUL bytes
            copy.write_row((sample["tenant"][i], sample["post_id"][i], sample["title"][i], body, vectors[i]))

    db.execute("create index tickets_hnsw on ops.tickets using hnsw (embedding vector_cosine_ops)")
    db.execute("create index tickets_tsv on ops.tickets using gin (tsv)")
    db.execute("create index tickets_tenant on ops.tickets (tenant_id)")
    db.execute("analyze ops.tickets")

    counts = db.execute("select tenant_id, count(*) from ops.tickets group by 1 order by 1").fetchall()
    print(counts, "load+index s", round(time.time() - start, 1))
