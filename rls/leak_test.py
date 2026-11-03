"""Cross-tenant leak test for ops.tickets (14 cases). Run apply_rls.py first.

A leak is any returned row whose tenant differs from the tenant bound for the transaction. Pass = 0 leaked rows and every expectation met.

  python leak_test.py                      -> results.json, exit 0 when all 14 pass
  python leak_test.py --positive-control   -> switches RLS off on the table, runs the same cases (many must FAIL, which proves the test can fail),
                                              then switches RLS back on. Writes results-positive-control.json; exit 0 when it did fail as expected.
"""
import json
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from common import GUC, LOADER, READER, bind, connect

HERE = Path(__file__).parent
POSITIVE_CONTROL = "--positive-control" in sys.argv
results, failed = {}, []


def check(name, ok, **info):
    results[name] = {"ok": bool(ok), **info}
    if not ok:
        failed.append(name)


def leaked(rows, tenant):
    return sum(1 for row in rows if row[0] != tenant)


def run_cases():
    with connect(autocommit=True) as admin:
        register_vector(admin)
        tenants = [row[0] for row in admin.execute("select distinct tenant_id from ops.tickets order by 1").fetchall()]
        own_rows = {t: admin.execute("select count(*) from ops.tickets where tenant_id=%s", (t,)).fetchone()[0] for t in tenants}
        # vectors and titles that exist only in the OTHER tenant, used as bait
        bait_vectors = {t: [r[0] for r in admin.execute("select embedding from ops.tickets where tenant_id<>%s limit 10", (t,)).fetchall()] for t in tenants}
        bait_titles = {t: [r[0] for r in admin.execute("select title from ops.tickets where tenant_id<>%s and length(title) >= 40 limit 20", (t,)).fetchall()] for t in tenants}
        sample_row = admin.execute("select tenant_id, post_id, title, body, embedding from ops.tickets where tenant_id=%s limit 1", (tenants[1],)).fetchone()

    check("0_two_tenants_found", len(tenants) >= 2, tenants=tenants)

    with connect(user=READER) as reader:
        register_vector(reader)
        role = reader.execute("select rolsuper, rolbypassrls from pg_roles where rolname=current_user").fetchone()
        check("0_role_not_superuser_not_bypassrls", role == (False, False), role=role)
        reader.rollback()

        for tenant in tenants:                                           # 1: only its own rows, and all of them
            rows = bind(reader, tenant).execute("select tenant_id from ops.tickets").fetchall()
            reader.rollback()
            check(f"1_as_{tenant}_sees_only_own_and_all_own", leaked(rows, tenant) == 0 and len(rows) == own_rows[tenant],
                  rows=len(rows), expected=own_rows[tenant], leaked=leaked(rows, tenant))

        for label, tenant in (("unbound", None), ("empty", ""), ("whitespace", " ")):   # 2: a missing or blank tenant sees nothing
            rows = bind(reader, tenant).execute("select tenant_id from ops.tickets").fetchall()
            reader.rollback()
            check(f"2_{label}_sees_nothing", len(rows) == 0, rows=len(rows))

        for tenant in tenants:                                           # 3: nearest neighbours of vectors taken from the other tenant
            leak = got = 0
            for vector in bait_vectors[tenant]:
                rows = bind(reader, tenant).execute("select tenant_id from ops.tickets order by embedding <=> %s limit 10", (vector,)).fetchall()
                reader.rollback()
                leak += leaked(rows, tenant)
                got += len(rows)
            check(f"3_vector_seeded_from_other_tenant_as_{tenant}", leak == 0, queries=len(bait_vectors[tenant]), rows=got, leaked=leak)

        for tenant in tenants:                                           # 4: titles that exist only in the other tenant, searched as text
            leak = hits = 0
            for title in bait_titles[tenant]:
                rows = bind(reader, tenant).execute("select tenant_id from ops.tickets where title = %s", (title,)).fetchall()
                reader.rollback()
                leak += leaked(rows, tenant)
                hits += len(rows)
            check(f"4_titles_of_other_tenant_as_{tenant}", leak == 0, titles=len(bait_titles[tenant]), rows_found=hits, leaked=leak)

        # 5: the caller adds its own WHERE tenant_id = <other tenant>
        rows = bind(reader, tenants[0]).execute("select tenant_id from ops.tickets where tenant_id=%s", (tenants[1],)).fetchall()
        reader.rollback()
        check("5_caller_asks_for_other_tenant", len(rows) == 0, rows=len(rows))

        # 9: connection reuse: tenant B in one transaction must not stick to the next transaction for tenant A
        bind(reader, tenants[1]).execute("select count(*) from ops.tickets")
        reader.commit()
        rows = bind(reader, tenants[0]).execute("select tenant_id from ops.tickets limit 1000").fetchall()
        reader.rollback()
        check("9_connection_reuse_set_local", leaked(rows, tenants[0]) == 0, rows=len(rows))

        # a finding, not a pass: an application that may call set_config can pick any tenant
        reader.cursor().execute("SELECT set_config(%s,%s,false)", (GUC, tenants[1]))
        seen = reader.execute("select count(*) from ops.tickets").fetchone()[0]
        reader.rollback()
        results["10_app_can_set_its_own_tenant"] = {"finding": True, "rows_seen_as_other_tenant": seen,
                                                    "note": "RLS guards against application bugs, not a malicious app: bind the tenant in a trusted layer"}

    # 6: the loader role tries to write a row that belongs to the other tenant
    tenant_of_row, post_id, title, body, embedding = sample_row
    with connect(user=LOADER) as loader:
        register_vector(loader)
        try:
            bind(loader, tenants[0]).execute(
                "insert into ops.tickets (tenant_id, post_id, title, body, embedding) values (%s, %s, %s, %s, %s)", (tenant_of_row, -post_id, title, body, embedding))
            verdict, ok = "ACCEPTED (leak)", False
        except psycopg.errors.InsufficientPrivilege:
            verdict, ok = "rejected by RLS WITH CHECK", True
        except psycopg.Error as error:
            verdict, ok = f"inconclusive: {type(error).__name__}", False
        loader.rollback()
        check("6_cross_tenant_write", ok, result=verdict)


def main():
    if POSITIVE_CONTROL:
        with connect(autocommit=True) as admin:
            admin.execute("alter table ops.tickets disable row level security")
    try:
        run_cases()
    finally:
        if POSITIVE_CONTROL:
            with connect(autocommit=True) as admin:
                admin.execute("alter table ops.tickets enable row level security")

    total_leaked = sum(v.get("leaked", 0) for v in results.values() if isinstance(v, dict))
    cases = [k for k in results if k != "10_app_can_set_its_own_tenant"]
    results["_summary"] = {"cases": len(cases), "failed": len(failed), "failed_cases": failed, "leaked_rows_total": total_leaked, "rls_off": POSITIVE_CONTROL}
    (HERE / ("results-positive-control.json" if POSITIVE_CONTROL else "results.json")).write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results["_summary"]))

    if POSITIVE_CONTROL:
        raise SystemExit(0 if failed and total_leaked > 0 else 1)      # the control must fail, otherwise the test proves nothing
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
