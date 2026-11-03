-- Incremental re-embedding with a stream and a task (demo on 1,017 tickets).
-- Embeddings are computed locally, so Snowflake only keeps track of WHAT changed: the stream records changed
-- rows, the task copies them into a work queue, and f8_worker.py embeds the queued rows and writes the vectors back.
-- Run   : snow sql -c ops_bench -f snowflake/f8_stream_task.sql   (then f8_worker.py)
-- Note  : the task is created suspended and run once by hand with EXECUTE TASK; nothing runs on a schedule.
USE SCHEMA SERVERFAULT;

-- 1. a small demo copy (every 320th ticket) with a content hash, and its vectors
CREATE OR REPLACE TABLE tickets_inc AS SELECT id, title, question, answer, comments, ticket_text, SHA2(ticket_text) AS content_hash FROM tickets WHERE id % 320 = 0;
CREATE OR REPLACE TABLE ticket_vec_inc AS SELECT v.id, v.vec, t.content_hash FROM ticket_vec_e5full256 v JOIN tickets_inc t ON t.id = v.id;

-- 2. the stream (what changed), the queue (what to re-embed) and the task that fills the queue when the stream has data
CREATE OR REPLACE STREAM tickets_inc_stream ON TABLE tickets_inc;
CREATE OR REPLACE TABLE reembed_queue (id NUMBER, action STRING, content_hash STRING, queued_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(), done_at TIMESTAMP_LTZ);
-- an UPDATE appears in a stream as a DELETE plus an INSERT; only the INSERT rows are the new versions, so only they are queued
-- the task runs on the warehouse of the current connection (a task needs a literal warehouse name, hence EXECUTE IMMEDIATE)
EXECUTE IMMEDIATE $$
BEGIN
  LET wh STRING := CURRENT_WAREHOUSE();
  EXECUTE IMMEDIATE 'CREATE OR REPLACE TASK enqueue_changed WAREHOUSE = ' || :wh || ' SCHEDULE = ''1 MINUTE'' WHEN SYSTEM$STREAM_HAS_DATA(''tickets_inc_stream'') AS INSERT INTO reembed_queue (id, action, content_hash) SELECT id, METADATA$ACTION, content_hash FROM tickets_inc_stream WHERE METADATA$ACTION = ''INSERT''';
END;
$$;
SELECT count(*) AS rows_in_demo_table FROM tickets_inc;

-- 3. change 10 rows (edit text and hash), look at the stream, then run the task body once by hand
UPDATE tickets_inc SET ticket_text = ticket_text || ' [edited]', content_hash = SHA2(ticket_text || ' [edited]') WHERE id IN (SELECT id FROM tickets_inc ORDER BY id LIMIT 10);
SELECT count(*) AS stream_rows FROM tickets_inc_stream;
EXECUTE TASK enqueue_changed;
