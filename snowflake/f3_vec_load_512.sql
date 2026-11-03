-- Put the two 512-token e5 vector sets into Snowflake VECTOR columns.
-- Input : emb_e5base_tq512_clean (title+question @512) and emb_e5base_full512 (full ticket @512) from f3_embed.py.
-- Output: ticket_vec_e5tq512 / query_vec_e5tq512 and ticket_vec_e5full512 / query_vec_e5full512 (same layout as f3_vec_load.sql).
-- Run   : snow sql -c ops_bench -f snowflake/f3_vec_load_512.sql -D bench_dir=$OPS_BENCH_DIR
USE SCHEMA SERVERFAULT;

CREATE OR REPLACE STAGE vec_stage_e5tq512 FILE_FORMAT = pq;
PUT file://<% bench_dir %>/sf/emb_e5base_tq512_clean/tickets.parquet @vec_stage_e5tq512/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://<% bench_dir %>/sf/emb_e5base_tq512_clean/q_clean.parquet @vec_stage_e5tq512/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://<% bench_dir %>/sf/emb_e5base_tq512_clean/q_raw.parquet @vec_stage_e5tq512/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
CREATE OR REPLACE TABLE ticket_vec_e5tq512 (id NUMBER, vec VECTOR(FLOAT, 768), src_file STRING);
COPY INTO ticket_vec_e5tq512 FROM (SELECT $1:id, $1:vec::ARRAY::VECTOR(FLOAT,768), METADATA$FILENAME FROM @vec_stage_e5tq512/tickets.parquet);
CREATE OR REPLACE TABLE query_vec_e5tq512 (qid NUMBER, variant STRING, vec VECTOR(FLOAT, 768));
COPY INTO query_vec_e5tq512 FROM (SELECT $1:id, 'clean', $1:vec::ARRAY::VECTOR(FLOAT,768) FROM @vec_stage_e5tq512/q_clean.parquet);
COPY INTO query_vec_e5tq512 FROM (SELECT $1:id, 'raw', $1:vec::ARRAY::VECTOR(FLOAT,768) FROM @vec_stage_e5tq512/q_raw.parquet);
SELECT 'e5tq512' arm, (SELECT count(*) FROM ticket_vec_e5tq512) tickets, (SELECT count(*) FROM query_vec_e5tq512) query_vectors;

CREATE OR REPLACE STAGE vec_stage_e5full512 FILE_FORMAT = pq;
PUT file://<% bench_dir %>/sf/emb_e5base_full512/tickets.parquet @vec_stage_e5full512/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://<% bench_dir %>/sf/emb_e5base_full512/q_clean.parquet @vec_stage_e5full512/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://<% bench_dir %>/sf/emb_e5base_full512/q_raw.parquet @vec_stage_e5full512/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
CREATE OR REPLACE TABLE ticket_vec_e5full512 (id NUMBER, vec VECTOR(FLOAT, 768), src_file STRING);
COPY INTO ticket_vec_e5full512 FROM (SELECT $1:id, $1:vec::ARRAY::VECTOR(FLOAT,768), METADATA$FILENAME FROM @vec_stage_e5full512/tickets.parquet);
CREATE OR REPLACE TABLE query_vec_e5full512 (qid NUMBER, variant STRING, vec VECTOR(FLOAT, 768));
COPY INTO query_vec_e5full512 FROM (SELECT $1:id, 'clean', $1:vec::ARRAY::VECTOR(FLOAT,768) FROM @vec_stage_e5full512/q_clean.parquet);
COPY INTO query_vec_e5full512 FROM (SELECT $1:id, 'raw', $1:vec::ARRAY::VECTOR(FLOAT,768) FROM @vec_stage_e5full512/q_raw.parquet);
SELECT 'e5full512' arm, (SELECT count(*) FROM ticket_vec_e5full512) tickets, (SELECT count(*) FROM query_vec_e5full512) query_vectors;
