"""Run a command with the Snowflake settings of one connection in its environment, so dbt and Dagster need no secrets in files.

Reads ~/.snowflake/config.toml, connection $SNOW_CONNECTION (default: ops_bench) and sets SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER,
SNOWFLAKE_TOKEN and, when the connection has them, SNOWFLAKE_ROLE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE.
Nothing is printed; the token is read from the file named by token_file_path (or the token key) and passed only to the child process.
usage: with_snowflake_env.py CMD [ARGS...]"""
import os, sys, tomllib, subprocess
c = tomllib.load(open(os.path.expanduser('~/.snowflake/config.toml'), 'rb'))['connections'][os.environ.get('SNOW_CONNECTION', 'ops_bench')]
tok = open(os.path.expanduser(c['token_file_path'])).read().strip() if 'token_file_path' in c else c['token']
env = dict(os.environ, SNOWFLAKE_ACCOUNT=c['account'], SNOWFLAKE_USER=c['user'], SNOWFLAKE_TOKEN=tok)
for key in ('role', 'warehouse', 'database'):
    if key in c:
        env['SNOWFLAKE_' + key.upper()] = c[key]
sys.exit(subprocess.run(sys.argv[1:], env=env).returncode)
