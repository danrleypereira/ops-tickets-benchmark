"""Streamlit in Snowflake: pick a held-out question, see the 5 closest tickets by exact vector search (SQL, no index).

A free-text search box is not possible here: turning a new sentence into a vector needs an embedding function, and the managed one
(AI_EMBED) is blocked on the trial account; the question vectors were computed locally. So the app offers the 30 held-out
questions that have a stored vector. Read-only.
"""
import streamlit as st
from snowflake.snowpark.context import get_active_session

session = get_active_session()
st.title('Find the earlier ticket that already solved this')
st.caption('Exact cosine search over 320,296 serverfault tickets, in SQL on an X-Small warehouse (public data, CC BY-SA 4.0).')

questions = session.sql(
    "SELECT qid, LEFT(query_clean, 140) AS question FROM SERVERFAULT.gold_queries WHERE split = 'heldout' ORDER BY qid LIMIT 30"
).to_pandas()
choice = st.selectbox('Question', questions['QID'], format_func=lambda qid: questions.loc[questions['QID'] == qid, 'QUESTION'].iloc[0].replace('\n', ' '))

if st.button('Search'):
    top = session.sql(f"""
        SELECT t.id AS ticket, ROUND(VECTOR_COSINE_SIMILARITY(q.vec, v.vec), 4) AS similarity, LEFT(t.title, 90) AS title,
               ARRAY_CONTAINS(t.id::VARIANT, PARSE_JSON(g.original_ids)) AS is_the_right_answer
        FROM SERVERFAULT.gold_queries g
        JOIN SERVERFAULT.query_vec_e5full256 q ON q.qid = g.qid AND q.variant = 'clean'
        JOIN SERVERFAULT.ticket_vec_e5full256 v ON TRUE
        JOIN SERVERFAULT.tickets t ON t.id = v.id
        WHERE g.qid = {int(choice)}
        ORDER BY VECTOR_COSINE_SIMILARITY(q.vec, v.vec) DESC LIMIT 5""").to_pandas()
    st.dataframe(top, use_container_width=True)
