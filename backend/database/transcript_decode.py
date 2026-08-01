"""Failure policy for transcript_segments decoding.

``transcript_segments`` is persisted in one of two incompatible physical
encodings, chosen per user at write time from ``users.data_protection_level``:
an AES-encrypted hex string ('enhanced') or a raw zlib blob ('standard'). The
read path in ``database.conversations`` dispatches on the Python runtime type
to tell them apart.

Upstream, a decode failure there is logged and swallowed, substituting an empty
segment list. That keeps the product resilient, but it makes a corrupt or
mis-migrated transcript indistinguishable from a genuinely empty conversation:
nothing raises, nothing alerts, the conversation renders fine and is simply
blank.

That trade is wrong while migrating the field to a different datastore, where a
silent bug can lose captures unnoticed. This module owns the choice so the
decode call sites stay declarative and the policy has one place to change.
"""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

_TRUTHY = {'1', 'true', 'yes', 'on'}


def strict_transcript_decode() -> bool:
    """Whether a decode failure should raise instead of yielding an empty list.

    Off by default, preserving upstream behaviour. Read at call time rather than
    import time so tests and runbooks can toggle it without reimporting.
    """
    return os.getenv('STRICT_TRANSCRIPT_DECODE', '').lower() in _TRUTHY


def handle_decode_failure(exc: Exception, uid: str, data: Dict[str, Any], encoding: str) -> None:
    """Log a transcript decode failure, and re-raise it when strict mode is on.

    ``encoding`` names the physical representation that failed ('zlib+aes' or
    'zlib'), which together with the conversation id is what you need to triage
    a bad row. A bare ``raise`` here re-raises the exception being handled by
    the caller's ``except`` block, preserving the original traceback.
    """
    logger.error(
        "transcript_segments decode failed (encoding=%s, uid=%s, conversation_id=%s): %s",
        encoding,
        uid,
        data.get('id'),
        exc,
    )
    if strict_transcript_decode():
        raise
