-- Live demo: exact vector search in Snowflake for ONE held-out question, over all 320,296 ticket vectors, in SQL.
-- What you see: the question, the 5 closest tickets (exact cosine, no index), whether the right original is among them, and the time.
-- The question is one where the system works (the right ticket comes first). Say so: over all 1,079 held-out questions the right ticket
-- is first for 13 out of 100 and in the top ten for 31 out of 100; this example was picked from the 19 short ones that come first.
-- Read-only (SELECT only), safe to run twice; cached results are switched off so the time is real. X-Small warehouse, about 5 s (measured 4.7 s and 4.2 s through the snow CLI).
-- Run : snow sql -c ops_bench -f snowflake/live_topk_one.sql
ALTER SESSION SET USE_CACHED_RESULT = FALSE;

SELECT g.qid, LEFT(g.query_clean, 200) AS question, g.original_ids AS right_answer_ids
FROM SERVERFAULT.gold_queries g WHERE g.qid = 225260;

SELECT t.id, ROUND(VECTOR_COSINE_SIMILARITY(q.vec, v.vec), 4) AS similarity, LEFT(t.title, 90) AS title,
       ARRAY_CONTAINS(t.id::VARIANT, PARSE_JSON(g.original_ids)) AS is_the_right_answer
FROM SERVERFAULT.gold_queries g
JOIN SERVERFAULT.query_vec_e5full256 q ON q.qid = g.qid AND q.variant = 'clean'
JOIN SERVERFAULT.ticket_vec_e5full256 v ON TRUE
JOIN SERVERFAULT.tickets t ON t.id = v.id
WHERE g.qid = 225260
ORDER BY VECTOR_COSINE_SIMILARITY(q.vec, v.vec) DESC
LIMIT 5;
