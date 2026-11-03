-- The big run: exact top-100 for all 2,158 held-out query vectors (clean + raw) in ONE statement on a LARGE warehouse.
-- It did not finish within the 90-minute budget and was cancelled by f4_snow_watchdog.py; f4_snow_scale.sql costs the work in smaller batches.
-- The last ALTER sets the warehouse back to X-Small so nothing keeps billing at the large size.
USE SCHEMA SERVERFAULT;
ALTER SESSION SET USE_CACHED_RESULT = FALSE;
SET wh = CURRENT_WAREHOUSE(); ALTER WAREHOUSE IDENTIFIER($wh) SET WAREHOUSE_SIZE = LARGE AUTO_SUSPEND = 60;
CREATE OR REPLACE TABLE snow_topk_e5full256 AS
WITH q AS (SELECT v.qid, v.variant, v.vec FROM query_vec_e5full256 v JOIN gold_queries g ON g.qid = v.qid WHERE g.split = 'heldout'),
s AS (SELECT q.qid, q.variant, t.id, VECTOR_COSINE_SIMILARITY(q.vec, t.vec) AS sim FROM q CROSS JOIN ticket_vec_e5full256 t)
SELECT qid, variant, id, sim, ROW_NUMBER() OVER (PARTITION BY qid, variant ORDER BY sim DESC) rn FROM s QUALIFY rn <= 100;
SET wh = CURRENT_WAREHOUSE(); ALTER WAREHOUSE IDENTIFIER($wh) SET WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60;
SELECT variant, count(*) FROM snow_topk_e5full256 GROUP BY 1;
