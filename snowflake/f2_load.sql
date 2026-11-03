-- Load the dba.stackexchange tickets and gold queries into Snowflake.
-- Input : tickets.parquet from f1_gold.py and gold_queries.parquet from f2_gold_to_parquet.py.
-- Output: tables TICKETS and GOLD_QUERIES in FORUM; METADATA$FILENAME is kept in src_file so every row
--         can be traced back to the file it came from (lineage).
-- Run   : snow sql -c ops_bench -f snowflake/f2_load.sql -D bench_dir=$OPS_BENCH_DIR

CREATE SCHEMA IF NOT EXISTS FORUM; USE SCHEMA FORUM;

-- 1. an internal stage for the files, then upload them (PUT compresses on the fly, but Parquet is already compressed)
CREATE OR REPLACE FILE FORMAT pq TYPE = PARQUET;
CREATE OR REPLACE STAGE forum_stage FILE_FORMAT = pq;
PUT file://<% bench_dir %>/gold/tickets.parquet @forum_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://<% bench_dir %>/gold/gold_queries.parquet @forum_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
LIST @forum_stage;

-- 2. tickets: one row per ticket, plus the source file and the load time
CREATE OR REPLACE TABLE tickets (id NUMBER, title STRING, tags STRING, created STRING, has_accepted BOOLEAN,
  question STRING, answer STRING, comments STRING, ticket_text STRING, src_file STRING, loaded_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP());
COPY INTO tickets (id, title, tags, created, has_accepted, question, answer, comments, ticket_text, src_file)
FROM (SELECT $1:id, $1:title, $1:tags, $1:created, $1:has_accepted, $1:question, $1:answer, $1:comments, $1:ticket_text, METADATA$FILENAME
      FROM @forum_stage/tickets.parquet) ;

-- 3. gold queries: one row per duplicate question, with the ids of its original(s) as JSON text
CREATE OR REPLACE TABLE gold_queries (qid NUMBER, original_ids STRING, query_raw STRING, query_clean STRING, leaked BOOLEAN,
  orig_has_accepted BOOLEAN, created STRING, split STRING);
COPY INTO gold_queries FROM (SELECT $1:qid, $1:original_ids, $1:query_raw, $1:query_clean, $1:leaked, $1:orig_has_accepted, $1:created, $1:split
      FROM @forum_stage/gold_queries.parquet);

-- 4. quick checks: row counts, and how many queries carried a duplicate banner (leaked) per split
SELECT 'tickets' t, count(*) n FROM tickets UNION ALL SELECT 'gold', count(*) FROM gold_queries;
SELECT split, count(*) n, sum(IFF(leaked,1,0)) leaked FROM gold_queries GROUP BY split;
