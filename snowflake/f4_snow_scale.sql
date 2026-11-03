-- Cost of exact vector search at two warehouse sizes.
--   1. LARGE: 200 held-out queries in one statement.
--   2. X-Small: a loop of 5 single-query statements, then the same 5 queries in one statement (loop vs batch).
-- The last SELECT lists the elapsed seconds of each statement; credits are then derived as seconds / 3600 x the warehouse rate (X-Small 1, Large 8 per hour).
USE SCHEMA SERVERFAULT;
ALTER SESSION SET USE_CACHED_RESULT = FALSE;
SET wh = CURRENT_WAREHOUSE(); ALTER WAREHOUSE IDENTIFIER($wh) SET WAREHOUSE_SIZE = LARGE AUTO_SUSPEND = 60;
CREATE OR REPLACE TABLE scale_large_200 AS /*SCALE_LARGE_200*/
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 200 OFFSET 0),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full256 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
SET wh = CURRENT_WAREHOUSE(); ALTER WAREHOUSE IDENTIFIER($wh) SET WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60;
CREATE OR REPLACE TABLE scale_loop_0 AS /*SCALE_LOOP_0*/
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 1 OFFSET 0),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full256 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
CREATE OR REPLACE TABLE scale_loop_1 AS /*SCALE_LOOP_1*/
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 1 OFFSET 1),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full256 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
CREATE OR REPLACE TABLE scale_loop_2 AS /*SCALE_LOOP_2*/
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 1 OFFSET 2),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full256 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
CREATE OR REPLACE TABLE scale_loop_3 AS /*SCALE_LOOP_3*/
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 1 OFFSET 3),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full256 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
CREATE OR REPLACE TABLE scale_loop_4 AS /*SCALE_LOOP_4*/
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 1 OFFSET 4),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full256 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
CREATE OR REPLACE TABLE scale_batch5_xs AS /*SCALE_BATCH5_XS*/
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 5 OFFSET 0),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full256 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
SELECT regexp_substr(query_text, 'SCALE_[A-Z0-9_]+') tag, round(total_elapsed_time/1000,1) sec, warehouse_size FROM table(information_schema.query_history_by_session(result_limit=>100)) WHERE query_text LIKE 'CREATE OR REPLACE TABLE scale_%' ORDER BY start_time;