-- one row per ticket, typed and trimmed; lineage columns kept
select id, title, tags, split_part(tags, ' ', 1) as category, try_to_timestamp(created) as created_at,
       has_accepted, length(question) as question_chars, length(answer) as answer_chars, src_file, loaded_at
from {{ source('serverfault', 'tickets') }}
