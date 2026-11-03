-- leakage and split summary of the duplicate gold set
select split, count(*) as queries, sum(iff(leaked, 1, 0)) as leaked_queries, round(100 * avg(iff(leaked, 1, 0)), 1) as leaked_pct,
       sum(iff(orig_has_accepted, 1, 0)) as originals_with_accepted_answer
from {{ source('serverfault', 'gold_queries') }} group by 1
