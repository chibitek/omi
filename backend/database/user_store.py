"""User-document storage, switchable between Firestore and the ``omi`` schema.

``database/users.py`` reaches for ``db.collection('users').document(uid)`` in
roughly twenty places and then calls ``.get()``, ``.set()`` or ``.update()`` on
it. That shape is Firestore's, not the domain's, so this module names the five
operations that actually exist and lets the storage behind them change.

WHY A SWITCH RATHER THAN A CUTOVER. The backend already boots and serves against
Postgres-less config -- every Firestore call is caught and logged rather than
fatal (verified: the container returns 200 on /v1/health with no GCP credentials
at all). So modules can move one at a time against a running service instead of
in a single cutover, and ``DB_BACKEND`` decides which store answers. Set it to
``postgres`` once a module's data has been migrated; leave it ``firestore``
otherwise. Rollback is an environment variable.

THE COLUMN/JSONB SPLIT. ``omi.users`` promotes exactly three fields to real
columns -- ``data_protection_level`` because it selects the transcript codec on
every conversation write, plus ``name`` and ``stripe_customer_id`` because they
are the only two fields any query filters on. Everything else keeps its document
shape in ``data``. Callers should not have to know that, so reads merge the
columns back into one dict and writes route each key to wherever it lives.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "PROMOTED_COLUMNS",
    "user_delete",
    "user_exists",
    "user_get",
    "user_set",
    "user_update",
    "uses_postgres",
]

# Fields that live in real columns on omi.users rather than inside `data`.
# Order is irrelevant; membership is what matters.
PROMOTED_COLUMNS = ("data_protection_level", "name", "stripe_customer_id")

_USERS_TABLE = "users"


def uses_postgres() -> bool:
    """Whether user documents are served from the omi schema.

    Read at call time, not import time, so tests and a rollback can flip it
    without reimporting the module graph.
    """
    return os.getenv("DB_BACKEND", "firestore").strip().lower() == "postgres"


def _split(payload: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Partition a document into (promoted column values, leftover jsonb)."""
    cols = {k: v for k, v in payload.items() if k in PROMOTED_COLUMNS}
    rest = {k: v for k, v in payload.items() if k not in PROMOTED_COLUMNS}
    return cols, rest


def user_get(uid: str) -> Optional[Dict[str, Any]]:
    """Return the user document, or None when the user does not exist.

    Firestore returned ``{}`` for a missing document in some call sites and
    ``None`` in others; this returns None and lets callers choose, which is why
    ``get_user_profile`` still coerces to ``{}`` itself.
    """
    if not uses_postgres():
        from ._client import db

        snapshot = db.collection('users').document(uid).get()
        return snapshot.to_dict() if snapshot.exists else None

    from . import _pg

    with _pg.connection() as conn:
        cur = conn.execute(
            f"select data, data_protection_level, name, stripe_customer_id "
            f"from {_pg.OMI_SCHEMA}.{_USERS_TABLE} where uid = %s",
            (uid,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    data, level, name, stripe_id = row
    doc: Dict[str, Any] = dict(data or {})
    # Columns win over any stale copy left inside the payload.
    doc["data_protection_level"] = level
    if name is not None:
        doc["name"] = name
    if stripe_id is not None:
        doc["stripe_customer_id"] = stripe_id
    return doc


def user_exists(uid: str) -> bool:
    if not uses_postgres():
        from ._client import db

        return bool(db.collection('users').document(uid).get().exists)

    from . import _pg

    with _pg.connection() as conn:
        cur = conn.execute(
            f"select 1 from {_pg.OMI_SCHEMA}.{_USERS_TABLE} where uid = %s",
            (uid,),
        )
        return cur.fetchone() is not None


def user_set(uid: str, payload: Dict[str, Any], merge: bool = False) -> None:
    """Create or replace the user document."""
    if not uses_postgres():
        from ._client import db

        db.collection('users').document(uid).set(payload, merge=merge)
        return

    from . import _pg
    from psycopg.types.json import Jsonb

    cols, rest = _split(payload)
    with _pg.connection() as conn:
        if merge:
            # Shallow merge on the payload; promoted columns overwrite only when
            # the caller actually supplied them, matching Firestore's merge.
            sets = [f"data = {_pg.OMI_SCHEMA}.{_USERS_TABLE}.data || excluded.data"]
            for c in cols:
                sets.append(f"{c} = coalesce(excluded.{c}, {_pg.OMI_SCHEMA}.{_USERS_TABLE}.{c})")
        else:
            sets = ["data = excluded.data"] + [f"{c} = excluded.{c}" for c in cols]
        sets.append("updated_at = now()")

        col_names = ["uid", "data"] + list(cols.keys())
        values: list[Any] = [uid, Jsonb(rest)] + [cols[c] for c in cols]
        placeholders = ", ".join(["%s"] * len(col_names))
        conn.execute(
            f"insert into {_pg.OMI_SCHEMA}.{_USERS_TABLE} ({', '.join(col_names)}) "
            f"values ({placeholders}) "
            f"on conflict (uid) do update set {', '.join(sets)}",
            values,
        )


def user_update(uid: str, patch: Dict[str, Any]) -> None:
    """Shallow-merge a patch into an existing user document.

    Mirrors Firestore's ``update()``: it does not create a missing row. Callers
    in users.py rely on that (they call it after an existence check, or on a
    document they just wrote).
    """
    if not uses_postgres():
        from ._client import db

        db.collection('users').document(uid).update(patch)
        return

    from . import _pg
    from psycopg.types.json import Jsonb

    cols, rest = _split(patch)
    assignments = ["updated_at = now()"]
    values: list[Any] = []
    if rest:
        assignments.append("data = data || %s")
        values.append(Jsonb(rest))
    for c, v in cols.items():
        assignments.append(f"{c} = %s")
        values.append(v)
    values.append(uid)

    with _pg.connection() as conn:
        conn.execute(
            f"update {_pg.OMI_SCHEMA}.{_USERS_TABLE} set {', '.join(assignments)} where uid = %s",
            values,
        )


def user_delete(uid: str) -> None:
    """Delete the user row. omi.people cascades via its foreign key."""
    if not uses_postgres():
        from ._client import db

        db.collection('users').document(uid).delete()
        return

    from . import _pg

    with _pg.connection() as conn:
        conn.execute(f"delete from {_pg.OMI_SCHEMA}.{_USERS_TABLE} where uid = %s", (uid,))
