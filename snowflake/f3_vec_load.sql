-- Put the local e5 vectors into Snowflake VECTOR columns.
-- Input : tickets.parquet, q_clean.parquet, q_raw.parquet from f3_embed.py (columns id, vec).
-- Output: ticket_vec_e5full256 (one vector per ticket) and query_vec_e5full256 (one per gold query and version, clean or raw).
-- Run   : snow sql -c ops_bench -f snowflake/f3_vec_load.sql -D bench_dir=$OPS_BENCH_DIR
USE SCHEMA SERVERFAULT;

-- 1. stage and upload
CREATE OR REPLACE STAGE vec_stage FILE_FORMAT = pq;
PUT file://<% bench_dir %>/sf/emb_e5base/tickets.parquet @vec_stage/e5full256/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://<% bench_dir %>/sf/emb_e5base/q_clean.parquet @vec_stage/e5full256/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://<% bench_dir %>/sf/emb_e5base/q_raw.parquet @vec_stage/e5full256/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- 2. ticket vectors; the Parquet list becomes an ARRAY and then a typed VECTOR(FLOAT, 768) while copying
CREATE OR REPLACE TABLE ticket_vec_e5full256 (id NUMBER, vec VECTOR(FLOAT, 768), src_file STRING);
COPY INTO ticket_vec_e5full256 FROM (SELECT $1:id, $1:vec::ARRAY::VECTOR(FLOAT,768), METADATA$FILENAME FROM @vec_stage/e5full256/tickets.parquet);

-- 3. query vectors, both versions in one table (variant says which)
CREATE OR REPLACE TABLE query_vec_e5full256 (qid NUMBER, variant STRING, vec VECTOR(FLOAT, 768));
COPY INTO query_vec_e5full256 FROM (SELECT $1:id, 'clean', $1:vec::ARRAY::VECTOR(FLOAT,768) FROM @vec_stage/e5full256/q_clean.parquet);
COPY INTO query_vec_e5full256 FROM (SELECT $1:id, 'raw', $1:vec::ARRAY::VECTOR(FLOAT,768) FROM @vec_stage/e5full256/q_raw.parquet);

-- 4. checks
SELECT count(*) FROM ticket_vec_e5full256; SELECT variant, count(*) FROM query_vec_e5full256 GROUP BY 1;
