"""Datetime-preserving JSON codec for documents stored as JSONB.

Firestore stores timestamps natively: you write a ``datetime`` and you read a
``datetime`` back. JSONB has no timestamp type, and psycopg refuses the value
outright -- ``TypeError: Object of type datetime is not JSON serializable`` --
so several user-document writers (``set_byok_active``,
``set_user_cancellation_feedback``, the deletion-wipe markers) would simply
crash on the Postgres backend.

Serialising to a string fixes the crash and introduces a worse bug. Call sites
type-check what they read:

    if isinstance(last_seen, datetime): ...
    else: return False          # database/users.py, is_byok_active

Store an ISO string without reviving it and BYOK reports every user inactive
forever. Nothing raises, nothing logs -- exactly the silent drift this migration
keeps having to design against.

So: encode ``datetime`` as RFC 3339 on write, and on read revive strings that
match a deliberately strict pattern -- date, ``T`` separator, time, and an
explicit offset or ``Z``. A naive-looking timestamp is left as a string, because
Firestore timestamps were always timezone-aware and a naive one did not come
from here.

The trade is a false positive: a genuine user string shaped exactly like an
RFC 3339 timestamp becomes a ``datetime``. Chosen deliberately over a sentinel
wrapper (``{"__datetime__": ...}``), which would be unambiguous but would make
every stored document unreadable by plain SQL and by anything else that touches
the row. Fidelity for the common case beats purity for a case that does not
occur.
"""

from __future__ import annotations

import re
from datetime import datetime, date
from typing import Any

__all__ = ["decode_document", "encode_document", "json_default"]

# Date, T, time, then a mandatory offset or Z. Deliberately strict: no bare
# dates, no naive datetimes, no space separator.
_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$")


def json_default(value: Any) -> Any:
    """``json.dumps(default=...)`` hook for types JSONB cannot hold."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        # Firestore held small blobs; hex keeps them round-trippable as text.
        return bytes(value).hex()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def encode_document(value: Any) -> Any:
    """Recursively convert datetimes to RFC 3339 strings for storage."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: encode_document(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_document(v) for v in value]
    return value


def decode_document(value: Any) -> Any:
    """Recursively revive RFC 3339 strings back into aware datetimes."""
    if isinstance(value, str):
        if _RFC3339.match(value):
            try:
                # fromisoformat handles 'Z' from Python 3.11 onward.
                return datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
            except ValueError:
                return value
        return value
    if isinstance(value, dict):
        return {k: decode_document(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_document(v) for v in value]
    return value
