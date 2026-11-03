"""Row-level security on ops.tickets, plus the two application roles.

- NOSUPERUSER NOBYPASSRLS roles: a superuser (or the table owner without FORCE) skips every policy, so the application must never connect as one.
- ENABLE + FORCE ROW LEVEL SECURITY: FORCE makes the policy apply to the table owner too.
- One policy, for reading and writing: the row's tenant must equal the tenant bound for the current transaction.
  `current_setting(..., true)` returns NULL when nothing is bound, and `tenant_id = NULL` is never true, so an unbound connection sees nothing.
"""
import json

from common import GUC, LOADER, READER, connect

with connect(autocommit=True) as db:
    for role in (READER, LOADER):
        if not db.execute("select 1 from pg_roles where rolname=%s", (role,)).fetchone():
            db.execute(f"create role {role} login nosuperuser nobypassrls nocreaterole nocreatedb")
    db.execute(f"grant usage on schema ops to {READER}, {LOADER}")
    db.execute("alter table ops.tickets enable row level security")
    db.execute("alter table ops.tickets force row level security")
    db.execute("drop policy if exists tenant_isolation on ops.tickets")
    db.execute(f"create policy tenant_isolation on ops.tickets using (tenant_id = current_setting('{GUC}', true)) "
               f"with check (tenant_id = current_setting('{GUC}', true))")
    db.execute(f"grant select on ops.tickets to {READER}")
    db.execute(f"grant select, insert, update, delete on ops.tickets to {LOADER}")
    db.execute(f"grant usage, select on all sequences in schema ops to {LOADER}")
    roles = db.execute("select rolname, rolsuper, rolbypassrls from pg_roles where rolname in (%s, %s) order by 1", (READER, LOADER)).fetchall()
print(json.dumps({"policy": "tenant_isolation on ops.tickets", "roles": roles}))
