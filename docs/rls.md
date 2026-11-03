# Tenant isolation with row-level security (adds `rls/`)
**What it is.** Two real tenants (Server Fault and Database Administrators questions) in one pgvector table; the application role cannot see the other tenant's rows even when a query forgets its `WHERE`. See `rls/README.md` for the setup and commands.

**Result** (one run, one host, 40,000 rows, pgvector 0.8): 14 of 14 leak-test cases pass with 0 leaked rows. With RLS switched off (positive control) 12 of 14 cases fail and 40,226 rows leak, so the test is not vacuous. Without any filter, 7.25% of a dba question's ten nearest neighbours are Server Fault posts (10.4% the other way round); the RLS reader returned 0 foreign rows for the same queries.

**Limits.** The application can set its own tenant: RLS protects against application bugs, not a malicious application; bind the tenant in a trusted layer. An approximate index with a filter can return fewer than ten rows (lost recall, not a leak).
