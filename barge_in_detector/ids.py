"""Row identifiers, as specified in docs/data-format.md#identifiers.

UUIDv7 for rows the project creates; UUIDv5 in a fixed namespace for rows that
must get the same ID whenever they are computed.
"""

import uuid
from collections.abc import Mapping

# Never change this: every derived ID depends on it.
NAMESPACE = uuid.UUID("3de823b9-152c-4f71-899d-8d2bde9cc65c")


def new_id() -> uuid.UUID:
    """A time-ordered UUIDv7 for a created row."""
    return uuid.uuid7()


def utterance_combination_id(spec: Mapping[str, str]) -> uuid.UUID:
    """ID of a criterion, from its factor values only. A factor at `default` is
    left out, so adding a factor whose criteria all take `default` keeps their
    IDs. The name prefix keeps these IDs apart from the retired conversation
    track's criteria."""
    pairs = ";".join(
        f"{factor}={spec[factor]}" for factor in sorted(spec) if spec[factor] != "default"
    )
    return uuid.uuid5(NAMESPACE, f"utterance_criterion:{pairs}")
