# dbt tests and a Dagster asset graph (adds `dbt/`, `dagster/`, `snowflake/with_snowflake_env.py`)
**Needs a Snowflake account** (dbt-snowflake): both read the credentials of your own connection through `snowflake/with_snowflake_env.py`; nothing secret is in the repository.

**What it is.** A dbt project over the tickets loaded into Snowflake: 3 views (`stg_tickets`, `ticket_stats_by_year`, `gold_quality`) and 6 tests, including a singular test (`tags_not_empty`) that fails if more than 5% of tags are empty. A Dagster graph of 11 assets covers the whole pipeline; cheap steps run, heavy steps only verify their saved output (size, hash, headline numbers) so re-running never re-spends GPU time or credits.

**Result.** The data-quality gap this closes: the first parse of the dump left every ticket without tags, and the stock dbt `not_null` test does not notice, because empty strings are not null. On a clone with the tags blanked, `not_null` passed and `tags_not_empty` failed. Then `dbt run` built 3 of 3 models (2.5 s on an X-Small warehouse) and `dbt test` passed 6 of 6. One Dagster run materialised 11 of 11 assets in about 30 s.

**How to run.** `cd dbt && python -I ../snowflake/with_snowflake_env.py dbt run --profiles-dir . --project-dir .` (then `test`); `dagster asset materialize -f dagster/ops_assets.py --select "*"` with `dagster` installed.

**Limits.** Snowflake only; the Dagster graph also references the other approaches' saved outputs.
