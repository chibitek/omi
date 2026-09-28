"""Person-record storage, switchable between Firestore and the ``omi`` schema.

People are the speech-profile identities behind speaker attribution: a name, a
list of enrolled voice-sample paths, and a speaker embedding. In Firestore they
are a subcollection under each user; in Postgres they are ``omi.people``, keyed
``(uid, id)`` with a cascading foreign key to ``omi.users``.

Two behaviours from the Firestore implementation are load-bearing and preserved
here rather than tidied away:

``update_name`` returns False for a missing person instead of raising. Upstream
learned this the hard way -- a bare ``.update()`` raises ``NotFound``, which
surfaces as an HTTP 500 where the caller wants a 404. The Postgres path gets the
same contract from the affected-row count, which also closes the check-then-act
race the Firestore version has to catch an exception for.

``add_speech_sample`` is capped and must not exceed the cap under concurrency.
Firestore needed an explicit transaction for that; Postgres gets it from a
single conditional UPDATE, so the read-modify-write disappears rather than being
emulated.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "add_speech_sample",
    "create_person",
    "delete_person",
    "get_people",
    "get_people_by_ids",
    "get_person",
    "get_person_by_name",
    "update_name",
    "uses_postgres",
]

_PEOPLE_TABLE = "people"
# Columns promoted out of `data` on omi.people.
_PROMOTED = ("name", "deleted", "speech_samples")


def uses_postgres() -> bool:
    """Whether person records are served from the omi schema. Read at call time."""
    return os.getenv("DB_BACKEND", "firestore").strip().lower() == "postgres"


def _people_ref(uid: str):
    from ._client import db

    return db.collection('users').document(uid).collection('people')


def _row_to_person(row: Sequence[Any]) -> Dict[str, Any]:
    """Assemble a person dict from (id, name, deleted, speech_samples, data)."""
    from ._json_codec import decode_document

    person_id, name, deleted, samples, data = row
    doc: Dict[str, Any] = dict(decode_document(data or {}))
    doc["id"] = person_id
    doc["name"] = name
    doc["deleted"] = deleted
    doc["speech_samples"] = decode_document(samples or [])
    return doc


_SELECT = "select id, name, deleted, speech_samples, data"


def create_person(uid: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if not uses_postgres():
        _people_ref(uid).document(data['id']).set(data)
        return data

    from . import _pg
    from ._json_codec import encode_document
    from psycopg.types.json import Jsonb

    payload = dict(encode_document(data))
    person_id = payload.pop("id")
    name = payload.pop("name", "")
    deleted = bool(payload.pop("deleted", False))
    samples = payload.pop("speech_samples", [])

    with _pg.connection() as conn:
        conn.execute(
            f"insert into {_pg.OMI_SCHEMA}.{_PEOPLE_TABLE} "
            f"(uid, id, name, deleted, speech_samples, data) values (%s, %s, %s, %s, %s, %s) "
            f"on conflict (uid, id) do update set name = excluded.name, "
            f"deleted = excluded.deleted, speech_samples = excluded.speech_samples, "
            f"data = excluded.data",
            (uid, person_id, name, deleted, Jsonb(samples), Jsonb(payload)),
        )
    return data


def get_person(uid: str, person_id: str) -> Optional[Dict[str, Any]]:
    if not uses_postgres():
        doc = _people_ref(uid).document(person_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        data.setdefault('id', doc.id)
        return data

    from . import _pg

    with _pg.connection() as conn:
        cur = conn.execute(
            f"{_SELECT} from {_pg.OMI_SCHEMA}.{_PEOPLE_TABLE} where uid = %s and id = %s",
            (uid, person_id),
        )
        row = cur.fetchone()
    return _row_to_person(row) if row else None


def get_people(uid: str) -> List[Dict[str, Any]]:
    if not uses_postgres():
        result = []
        for person in _people_ref(uid).stream():
            data = person.to_dict()
            data.setdefault('id', person.id)
            result.append(data)
        return result

    from . import _pg

    with _pg.connection() as conn:
        cur = conn.execute(
            f"{_SELECT} from {_pg.OMI_SCHEMA}.{_PEOPLE_TABLE} where uid = %s and deleted = false",
            (uid,),
        )
        return [_row_to_person(r) for r in cur.fetchall()]


def get_person_by_name(uid: str, name: str) -> Optional[Dict[str, Any]]:
    if not uses_postgres():
        from google.cloud.firestore_v1 import FieldFilter

        docs = list(_people_ref(uid).where(filter=FieldFilter('name', '==', name)).limit(1).stream())
        if not docs:
            return None
        data = docs[0].to_dict()
        data.setdefault('id', docs[0].id)
        return data

    from . import _pg

    with _pg.connection() as conn:
        cur = conn.execute(
            f"{_SELECT} from {_pg.OMI_SCHEMA}.{_PEOPLE_TABLE} "
            f"where uid = %s and name = %s and deleted = false limit 1",
            (uid, name),
        )
        row = cur.fetchone()
    return _row_to_person(row) if row else None


def get_people_by_ids(uid: str, person_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Fetch people by id. Order is unspecified, matching Firestore's get_all()."""
    if not person_ids:
        return []

    if not uses_postgres():
        from ._client import db

        refs = [_people_ref(uid).document(pid) for pid in person_ids]
        out = []
        for doc in db.get_all(refs):
            if doc.exists:
                data = doc.to_dict()
                data.setdefault('id', doc.id)
                out.append(data)
        return out

    from . import _pg

    with _pg.connection() as conn:
        cur = conn.execute(
            f"{_SELECT} from {_pg.OMI_SCHEMA}.{_PEOPLE_TABLE} where uid = %s and id = any(%s)",
            (uid, list(person_ids)),
        )
        return [_row_to_person(r) for r in cur.fetchall()]


def update_name(uid: str, person_id: str, name: str) -> bool:
    """Rename a person. False when absent, so callers 404 rather than 500."""
    if not uses_postgres():
        from google.api_core.exceptions import NotFound

        person_ref = _people_ref(uid).document(person_id)
        if not person_ref.get().exists:
            return False
        try:
            person_ref.update({'name': name})
        except NotFound:
            return False
        return True

    from . import _pg

    # One statement, so there is no check-then-act window: the row count is the
    # existence answer.
    with _pg.connection() as conn:
        cur = conn.execute(
            f"update {_pg.OMI_SCHEMA}.{_PEOPLE_TABLE} set name = %s where uid = %s and id = %s",
            (name, uid, person_id),
        )
        return cur.rowcount > 0


def delete_person(uid: str, person_id: str) -> None:
    if not uses_postgres():
        _people_ref(uid).document(person_id).delete()
        return

    from . import _pg

    with _pg.connection() as conn:
        conn.execute(
            f"delete from {_pg.OMI_SCHEMA}.{_PEOPLE_TABLE} where uid = %s and id = %s",
            (uid, person_id),
        )


def add_speech_sample(uid: str, person_id: str, sample_path: str, max_samples: int) -> bool:
    """Append a sample path unless the person is missing or already at the cap.

    The cap is enforced inside the statement, so two concurrent callers cannot
    both observe ``len(samples) < max`` and both append. Firestore needed a
    transaction for this; here the condition is part of the UPDATE.
    """
    if not uses_postgres():
        raise NotImplementedError("Firestore path uses _add_sample_transaction in users.py")

    from . import _pg
    from psycopg.types.json import Jsonb

    with _pg.connection() as conn:
        cur = conn.execute(
            f"update {_pg.OMI_SCHEMA}.{_PEOPLE_TABLE} "
            f"set speech_samples = speech_samples || %s "
            f"where uid = %s and id = %s "
            f"  and jsonb_array_length(speech_samples) < %s "
            f"  and not (speech_samples @> %s)",
            (Jsonb([sample_path]), uid, person_id, max_samples, Jsonb([sample_path])),
        )
        return cur.rowcount > 0
