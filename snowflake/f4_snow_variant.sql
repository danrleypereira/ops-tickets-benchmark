-- Same probe as f4_snow_topk.sql, but the similarity is computed once (a CTE) and only 20 queries: 250 s = 12.5 s per query.
USE SCHEMA SERVERFAULT;
ALTER SESSION SET USE_CACHED_RESULT = FALSE;
CREATE OR REPLACE TABLE probe_topk_b AS
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 20),
s AS (SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full256 t)
SELECT qid, id, sim, ROW_NUMBER() OVER (PARTITION BY qid ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
SELECT 'secs' k, total_elapsed_time/1000 v FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION(RESULT_LIMIT=>3)) WHERE query_text ILIKE 'CREATE OR REPLACE TABLE probe_topk_b%' ORDER BY start_time DESC LIMIT 1;
