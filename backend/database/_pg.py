"""Postgres access layer for the Supabase-backed ``omi`` schema.

This replaces ``database/_client.py``'s Firestore singleton. It is deliberately
NOT Firestore-shaped.

WHY NOT A FIRESTORE-COMPATIBLE SHIM. The obvious move is a shim that keeps
``db.collection(...).document(...)`` call sites working, and it is the wrong one.
The Firestore client escapes the ``database/`` package in exactly two files, so a
shim protects no external call sites -- it only preserves the *internal* ones,
while forcing a 1,500-2,000 line emulation of chained ``.where(FieldFilter(...))``
builders, ``DocumentSnapshot`` semantics, 500-document batch chunking,
``ArrayUnion``/``Increment`` sentinels, ``collection_group`` and ``.count()``
aggregation, plus Firestore's undocumented edge behaviour. It would also
permanently foreclose the things Postgres is here for: joins, pgvector, RLS, and
writing OMI's action items straight into ``public.action_items``.

So this module exposes a small explicit API instead, and the callers that need
real relational shapes (conversations, users, people) get purpose-written SQL
rather than going through here.

WHAT THIS IS FOR. The ~29 low-traffic collections that are genuinely documents:
announcements, folders, goals, trends, notifications, staged tasks and friends.
They live in ``(uid, id, org_id, data jsonb)`` tables, and this is their door.

ONE THING THAT IS BETTER THAN FIRESTORE, NOT EMULATED. ``counter_add`` replaces
``firestore.Increment`` on dotted field paths (llm_usage, user_usage, fair_use)
with a real ``INSERT ... ON CONFLICT DO UPDATE``. Atomic, indexable, and
aggregatable in SQL, which the JSONB-sentinel version never was.

CONNECTION MODEL. Sync psycopg 3 with a pool, because the database layer is
synchronous; an async driver would mean rewriting every call site rather than the
storage behind them. The pool is lazy for the same reason the Firestore client
was: importing this module must never require credentials, or the whole app
becomes unimportable in tests and CI.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "OMI_SCHEMA",
    "connection",
    "counter_add",
    "doc_delete",
    "doc_get",
    "doc_set",
    "doc_update",
    "docs_batch_set",
    "docs_query",
    "get_pool",
    "reset_pool",
]

OMI_SCHEMA = "omi"

# Identifier allow-list. Collection names and field names are interpolated into
# SQL (they cannot be bound as parameters), so every one is validated against the
# set of characters a Postgres identifier may contain here. Anything else raises
# rather than reaching the database.
_IDENT_OK = set("abcdefghijklmnopqrstuvwxyz0123456789_")

_pool: Any = None
_pool_lock = Lock()


def _ident(name: str) -> str:
    """Validate an identifier before it is interpolated into SQL."""
    if not name or not set(name.lower()) <= _IDENT_OK or name[0].isdigit():
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name.lower()


def _dsn() -> str:
    dsn = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL") or ""
    if not dsn:
        raise RuntimeError("DATABASE_URL (or SUPABASE_DB_URL) is not set -- the omi schema lives in cloud Supabase.")
    if "localhost" in dsn or "127.0.0.1" in dsn:
        # A hosted service cannot reach a local database, and a local Postgres is
        # explicitly not part of this architecture. Fail loudly rather than
        # silently pointing at whatever happens to be listening.
        raise RuntimeError(f"DATABASE_URL points at a local host; the omi schema is cloud Supabase: {dsn!r}")
    return dsn


def get_pool() -> Any:
    """Return the process-wide connection pool, opening it on first use."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from psycopg_pool import ConnectionPool  # imported lazily: see module docstring

                _pool = ConnectionPool(
                    conninfo=_dsn(),
                    min_size=int(os.getenv("PG_POOL_MIN_SIZE", "1")),
                    max_size=int(os.getenv("PG_POOL_MAX_SIZE", "10")),
                    open=True,
                    kwargs={"options": f"-c search_path={OMI_SCHEMA},public"},
                )
                logger.info("omi postgres pool opened (schema=%s)", OMI_SCHEMA)
    return _pool


def reset_pool() -> None:
    """Close the pool. For tests and for a clean shutdown."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def connection() -> Iterator[Any]:
    """Borrow a pooled connection. Commits on success, rolls back on error."""
    with get_pool().connection() as conn:
        yield conn


def _row_to_doc(row: Optional[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    """Materialise a (id, data) row as the document dict callers expect.

    ``id`` is folded into the payload because Firestore exposed it on the
    snapshot rather than inside the document, and call sites do
    ``data.setdefault('id', doc.id)``.
    """
    if row is None:
        return None
    doc_id, data = row[0], (row[1] or {})
    doc = dict(data)
    doc.setdefault("id", doc_id)
    return doc


def doc_get(collection: str, uid: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one document, or None when it does not exist."""
    table = _ident(collection)
    with connection() as conn:
        cur = conn.execute(
            f"select id, data from {OMI_SCHEMA}.{table} where uid = %s and id = %s",
            (uid, doc_id),
        )
        return _row_to_doc(cur.fetchone())


def doc_set(collection: str, uid: str, doc_id: str, data: Dict[str, Any], merge: bool = False) -> None:
    """Create or replace a document.

    ``merge=True`` mirrors Firestore's ``set(..., merge=True)``: a shallow
    top-level merge, so absent keys are preserved rather than dropped.
    """
    table = _ident(collection)
    from psycopg.types.json import Jsonb

    payload = Jsonb(data)
    conflict = (
        f"update set data = {OMI_SCHEMA}.{table}.data || excluded.data" if merge else "update set data = excluded.data"
    )
    with connection() as conn:
        conn.execute(
            f"insert into {OMI_SCHEMA}.{table} (uid, id, data) values (%s, %s, %s) "
            f"on conflict (uid, id) do {conflict}",
            (uid, doc_id, payload),
        )


def doc_update(collection: str, uid: str, doc_id: str, patch: Dict[str, Any]) -> None:
    """Shallow-merge a patch into an existing document.

    Unlike ``doc_set(merge=True)`` this does not create the row, matching
    Firestore's ``update()``, which fails on a missing document.
    """
    table = _ident(collection)
    from psycopg.types.json import Jsonb

    with connection() as conn:
        conn.execute(
            f"update {OMI_SCHEMA}.{table} set data = data || %s where uid = %s and id = %s",
            (Jsonb(patch), uid, doc_id),
        )


def doc_delete(collection: str, uid: str, doc_id: str) -> None:
    table = _ident(collection)
    with connection() as conn:
        conn.execute(
            f"delete from {OMI_SCHEMA}.{table} where uid = %s and id = %s",
            (uid, doc_id),
        )


def docs_query(
    collection: str,
    uid: Optional[str] = None,
    *,
    eq: Optional[Dict[str, Any]] = None,
    in_: Optional[Dict[str, Sequence[Any]]] = None,
    gte: Optional[Dict[str, Any]] = None,
    lte: Optional[Dict[str, Any]] = None,
    lt: Optional[Dict[str, Any]] = None,
    contains: Optional[Dict[str, Any]] = None,
    order: Optional[Sequence[Tuple[str, str]]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Query documents.

    The operator set is exactly what the Firestore call sites actually use --
    measured across ``database/*.py``: ``==`` (112), ``in`` (16), ``>=`` (14),
    ``<=`` (9), ``array_contains`` (6) and ``<`` (4). No ``!=``, no ``not-in``,
    no ``array_contains_any``, and no cursor pagination anywhere, which is why
    offset/limit is sufficient here.

    ``uid=None`` drops the user predicate, which is how the five
    ``collection_group`` call sites are served: in SQL a collection-group query
    is just the same query without the partition key.
    """
    table = _ident(collection)
    where: List[str] = []
    params: List[Any] = []

    if uid is not None:
        where.append("uid = %s")
        params.append(uid)

    def _add(clauses: Optional[Dict[str, Any]], sql_op: str) -> None:
        for field, value in (clauses or {}).items():
            where.append(f"data ->> %s {sql_op} %s")
            params.extend([_ident(field), value if isinstance(value, str) else str(value)])

    _add(eq, "=")
    _add(gte, ">=")
    _add(lte, "<=")
    _add(lt, "<")

    for field, values in (in_ or {}).items():
        seq = list(values)
        if not seq:
            # Firestore rejects an empty `in`; an empty result is the honest
            # translation and avoids emitting `in ()`, which is a syntax error.
            return []
        where.append("data ->> %s = any(%s)")
        params.extend([_ident(field), [str(v) for v in seq]])

    for field, value in (contains or {}).items():
        # array_contains: the JSONB array at `field` contains `value`.
        from psycopg.types.json import Jsonb

        where.append("data -> %s @> %s")
        params.extend([_ident(field), Jsonb([value])])

    sql = [f"select id, data from {OMI_SCHEMA}.{table}"]
    if where:
        sql.append("where " + " and ".join(where))

    if order:
        parts = []
        for field, direction in order:
            desc = str(direction).lower() in ("desc", "descending")
            parts.append(f"data ->> '{_ident(field)}' {'desc' if desc else 'asc'}")
        sql.append("order by " + ", ".join(parts))

    if limit is not None:
        sql.append("limit %s")
        params.append(int(limit))
    if offset:
        sql.append("offset %s")
        params.append(int(offset))

    with connection() as conn:
        cur = conn.execute(" ".join(sql), params)
        return [doc for doc in (_row_to_doc(r) for r in cur.fetchall()) if doc is not None]


def docs_batch_set(collection: str, uid: str, rows: Iterable[Tuple[str, Dict[str, Any]]]) -> int:
    """Upsert many documents in one statement.

    Replaces the 30-plus ``db.batch()`` sites with their manual 500-document
    chunking; Postgres has no such per-batch ceiling, so the chunking goes away
    with the emulation.
    """
    table = _ident(collection)
    from psycopg.types.json import Jsonb

    payload = [(uid, doc_id, Jsonb(data)) for doc_id, data in rows]
    if not payload:
        return 0
    with connection() as conn, conn.cursor() as cur:
        cur.executemany(
            f"insert into {OMI_SCHEMA}.{table} (uid, id, data) values (%s, %s, %s) "
            f"on conflict (uid, id) do update set data = excluded.data",
            payload,
        )
    return len(payload)


def counter_add(table: str, keys: Dict[str, Any], counters: Dict[str, int]) -> None:
    """Atomically add to one or more counter columns.

    Replaces ``firestore.Increment`` on dotted field paths such as
    ``f"{feature}.{model}.input_tokens"``. Emulating that with ``jsonb_set``
    would be strictly worse than what Postgres offers, so these collections get
    real counter tables with real columns and this does an upsert-and-add.
    """
    tbl = _ident(table)
    key_cols = [_ident(k) for k in keys]
    cnt_cols = [_ident(c) for c in counters]
    if not cnt_cols:
        return

    cols = key_cols + cnt_cols
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c} = {tbl}.{c} + excluded.{c}" for c in cnt_cols)
    params = [keys[k] for k in keys] + [int(counters[c]) for c in counters]

    with connection() as conn:
        conn.execute(
            f"insert into {OMI_SCHEMA}.{tbl} ({', '.join(cols)}) values ({placeholders}) "
            f"on conflict ({', '.join(key_cols)}) do update set {updates}",
            params,
        )
