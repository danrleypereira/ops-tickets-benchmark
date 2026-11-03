-- Batched exact top-100 in Snowflake for the 512-token vector sets: 100 held-out clean queries per statement on X-Small.
-- Cost per 1,000 queries = elapsed seconds / 100 x 1000 / 3600 x 1 credit per hour.
USE SCHEMA SERVERFAULT;
SET wh = CURRENT_WAREHOUSE(); ALTER WAREHOUSE IDENTIFIER($wh) SET WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60;
ALTER SESSION SET USE_CACHED_RESULT = FALSE;
CREATE OR REPLACE TABLE topk100_e5tq512 AS /*TOPK100_E5TQ512*/
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5tq512 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 100),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5tq512 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
CREATE OR REPLACE TABLE topk100_e5full512 AS /*TOPK100_E5FULL512*/
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full512 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 100),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full512 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
SELECT regexp_substr(query_text, 'TOPK100_[A-Z0-9]+') tag, round(total_elapsed_time/1000,1) sec, warehouse_size FROM table(information_schema.query_history_by_session(result_limit=>50)) WHERE query_text LIKE 'CREATE OR REPLACE TABLE topk100_%' ORDER BY start_time;
