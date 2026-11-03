"""Step 3: why the leak test is not vacuous on real data.

For 200 questions per tenant (the question's own embedding as the query, itself excluded) it compares two readers:
  (a) a superuser with no tenant filter: what share of the top-10 neighbours belongs to the OTHER site. This is what a missing filter would leak.
  (b) the RLS reader bound to the tenant: the same query returns only its own rows.

Output: results-overlap.json next to this file.
"""
import json
from pathlib import Path

import numpy as np
from pgvector.psycopg import register_vector

from common import READER, bind, connect

QUERIES_PER_TENANT = 200
results = {}

with connect(autocommit=True) as superuser:
    register_vector(superuser)
    rng = np.random.default_rng(9)

    for tenant, other_site in (("dba", "serverfault"), ("serverfault", "dba")):
        ticket_ids = [row[0] for row in superuser.execute("select id from ops.tickets where tenant_id=%s", (tenant,)).fetchall()]
        picked = rng.choice(ticket_ids, QUERIES_PER_TENANT, replace=False).tolist()

        unfiltered_rows = foreign_rows = 0
        reader_rows = reader_leaked = 0

        with connect(user=READER) as reader:
            register_vector(reader)

            for ticket_id in picked:
                vector = superuser.execute("select embedding from ops.tickets where id=%s", (ticket_id,)).fetchone()[0]

                # (a) no tenant filter at all
                neighbours = superuser.execute("select id, tenant_id from ops.tickets where id<>%s order by embedding <=> %s limit 10", (ticket_id, vector)).fetchall()
                for _, owner in neighbours:
                    unfiltered_rows += 1
                    foreign_rows += owner == other_site

                # (b) the RLS reader; the tenant is bound per transaction, never per session
                with reader.transaction():
                    bind(reader, tenant)
                    rows = reader.execute("select tenant_id from ops.tickets order by embedding <=> %s limit 10", (vector,)).fetchall()
                    reader_rows += len(rows)
                    reader_leaked += sum(owner != tenant for (owner,) in rows)

        results[tenant] = {
            "queries": QUERIES_PER_TENANT,
            "unfiltered_top10_rows": unfiltered_rows,
            "share_from_other_site": round(foreign_rows / unfiltered_rows, 4),
            "rls_reader_rows": reader_rows,
            "rls_reader_leaked": reader_leaked,
        }

Path(__file__).with_name("results-overlap.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
