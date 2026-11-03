-- Timing probe for exact vector search in Snowflake: top-100 tickets for 50 held-out queries, one batched statement.
-- The similarity is computed twice here (in SELECT and in the window); f4_snow_variant.sql computes it once.
-- Result: 693 s on an X-Small warehouse = 13.9 s per query. Cached results are switched off so the timing is real.
USE SCHEMA SERVERFAULT;
ALTER SESSION SET USE_CACHED_RESULT = FALSE;
-- timing probe: 50 held-out clean queries against all 320,296 tickets
CREATE OR REPLACE TABLE probe_topk AS
WITH q AS (SELECT v.qid, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE v.variant = 'clean' AND g.split = 'heldout' ORDER BY v.qid LIMIT 50)
SELECT qid, id, sim, rn FROM (
  SELECT q.qid, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim,
         ROW_NUMBER() OVER (PARTITION BY q.qid ORDER BY VECTOR_COSINE_SIMILARITY(q.vec, t.vec) DESC) rn
  FROM q CROSS JOIN ticket_vec_e5full256 t) WHERE rn <= 100;
SELECT count(*) FROM probe_topk;
SELECT query_id, total_elapsed_time/1000 AS sec FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION(RESULT_LIMIT=>5)) WHERE query_text ILIKE 'CREATE OR REPLACE TABLE probe_topk%' ORDER BY start_time DESC LIMIT 1;
