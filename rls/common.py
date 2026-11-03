"""Shared connection helpers for the RLS demo (local throwaway container, trust auth, no passwords)."""
import os

import psycopg

HOST = "127.0.0.1"
PORT = int(os.environ.get("PG_PORT", 25433))
DB = os.environ.get("DB_NAME", "opsrls")
GUC = "app.tenant_id"           # the setting the policy reads
READER, LOADER = "rls_reader", "rls_loader"


def connect(user="postgres", dbname=None, autocommit=False):
    return psycopg.connect(host=HOST, port=PORT, user=user, dbname=dbname or DB, autocommit=autocommit)


def bind(conn, tenant):
    """Bind the tenant for THIS transaction only (set_config with is_local=true); never a session-level SET."""
    cur = conn.cursor()
    if tenant is not None:
        cur.execute("SELECT set_config(%s, %s, true)", (GUC, tenant))
    return cur
