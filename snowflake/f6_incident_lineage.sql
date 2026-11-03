-- A ServiceNow-style view and a lineage view over the serverfault tickets.
-- incident: the same tickets with the field names an IT service-management system uses (tags play the role of category / configuration item).
-- lineage : post id -> staged Parquet file -> ticket row -> vector file -> how often it was retrieved, so any answer can be traced to its source.
-- Needs the table scale_large_200 (top-100 results of the 200-query test in f4_snow_scale.sql).
-- Run   : snow sql -c ops_bench -f snowflake/f6_incident_lineage.sql
USE SCHEMA SERVERFAULT;

CREATE OR REPLACE VIEW incident AS
SELECT 'SF' || LPAD(id::STRING, 8, '0') AS number, title AS short_description, question AS description,
       comments AS work_notes, answer AS close_notes, SPLIT_PART(tags, ' ', 1) AS category, tags AS cmdb_ci,
       created AS opened_at, IFF(has_accepted, 'Resolved', 'Open') AS state, id AS source_post_id
FROM tickets;

CREATE OR REPLACE VIEW lineage AS
SELECT t.id AS post_id, t.src_file AS ticket_file, t.loaded_at AS ticket_loaded_at, v.src_file AS vector_file,
       (SELECT count(*) FROM scale_large_200 r WHERE r.id = t.id) AS times_in_top100_of_200_queries
FROM tickets t LEFT JOIN ticket_vec_e5full256 v ON v.id = t.id;

-- checks
SELECT number, short_description, category, state FROM incident LIMIT 3;

-- trace retrieved tickets end to end: result -> ticket -> parquet file -> vector file
SELECT r.qid, r.rn AS rank, l.post_id, l.ticket_file, l.vector_file, i.number, i.short_description
FROM scale_large_200 r JOIN lineage l ON l.post_id = r.id JOIN incident i ON i.source_post_id = r.id WHERE r.rn <= 2 ORDER BY r.qid, r.rn LIMIT 4;
SELECT count(*) AS tickets_with_lineage, count_if(ticket_file IS NOT NULL AND vector_file IS NOT NULL) AS with_both_files FROM lineage;
