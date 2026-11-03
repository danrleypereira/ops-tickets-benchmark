-- Optional: Streamlit in Snowflake app for the Snowsight moment. Creates a stage, uploads the app and registers it.
-- Run: snow sql -c ops_bench -f snowflake/streamlit/create_streamlit.sql -D repo_dir=$PWD      Drop: DROP STREAMLIT SERVERFAULT.ticket_search;
USE SCHEMA SERVERFAULT;
CREATE OR REPLACE STAGE streamlit_stage;
PUT file://<% repo_dir %>/snowflake/streamlit/streamlit_app.py @streamlit_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
EXECUTE IMMEDIATE $$
BEGIN
  LET wh STRING := CURRENT_WAREHOUSE();
  EXECUTE IMMEDIATE 'CREATE OR REPLACE STREAMLIT ticket_search ROOT_LOCATION = ''@SERVERFAULT.streamlit_stage'' MAIN_FILE = ''streamlit_app.py'' QUERY_WAREHOUSE = ' || :wh;
END;
$$;
SHOW STREAMLITS IN SCHEMA SERVERFAULT;
