select year(created_at) as year, count(*) as tickets, avg(iff(has_accepted, 1, 0)) as accepted_share, median(question_chars) as median_question_chars
from {{ ref('stg_tickets') }} group by 1 order by 1
