-- the check that would have caught the empty-tags parser bug: not_null alone passes on empty strings
select bad, n from (select count_if(tags = '') as bad, count(*) as n from {{ ref('stg_tickets') }}) where bad > 0.05 * n
