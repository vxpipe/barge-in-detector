"""Row identifiers, as specified in docs/data-format.md#identifiers.

UUIDv7 for rows the project creates; UUIDv5 in a fixed namespace for rows that
must get the same ID whenever they are computed.
"""

import uuid
from collections.abc import Iterable, Mapping

# Never change this: every derived ID depends on it.
NAMESPACE = uuid.UUID("3de823b9-152c-4f71-899d-8d2bde9cc65c")


def new_id() -> uuid.UUID:
    """A time-ordered UUIDv7 for a created row."""
    return uuid.uuid7()


def combination_id(spec: Mapping[str, str]) -> uuid.UUID:
    """ID of a criterion, from its spec factor values only.

    A factor at the value `default` is left out, so adding a factor whose
    existing criteria all take `default` keeps their IDs.
    """
    pairs = ";".join(
        f"{factor}={spec[factor]}" for factor in sorted(spec) if spec[factor] != "default"
    )
    return uuid.uuid5(NAMESPACE, f"criterion:{pairs}")


def utterance_combination_id(spec: Mapping[str, str]) -> uuid.UUID:
    """ID of an utterance-track criterion. The track is part of the name, so it
    never equals a conversation criterion's ID for the same factor values."""
    pairs = ";".join(
        f"{factor}={spec[factor]}" for factor in sorted(spec) if spec[factor] != "default"
    )
    return uuid.uuid5(NAMESPACE, f"utterance_criterion:{pairs}")


def split_group_id(combination_ids: Iterable[uuid.UUID]) -> uuid.UUID:
    """ID of a split group: derived from the smallest combination_id in it."""
    return uuid.uuid5(NAMESPACE, f"split_group:{min(combination_ids)}")


def example_id(conversation_id: uuid.UUID, *, turn_shape: int, prefix_index: int) -> uuid.UUID:
    """ID of an example; independent of builder and cleaner versions."""
    return uuid.uuid5(NAMESPACE, f"example:{conversation_id}:{turn_shape}:{prefix_index}")
