"""Transcript cleaner, version 1. Rules: docs/transcript-format.md.

The runtime (Vxpipe) must produce exactly the same output; the golden cases in
tests/fixtures/cleaner_golden.yaml define parity.
"""

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

CLEANER_VERSION = "1"

_APOSTROPHES = {"'", "’"}
_JOINERS = {"‌", "‍"}  # ZWNJ, ZWJ: needed in Indic scripts and Persian
# Unicode classes these as punctuation (Po), but they belong to the number.
_KEPT_PUNCTUATION = {"%", "‰"}


class EmptyTurnError(ValueError):
    """A turn has no text left after cleaning."""


@dataclass(frozen=True)
class CleanWindow:
    transcript: str
    speaker_labels: list[str]
    texts: list[str]


def _wordlike(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LMN"


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    out: list[str] = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        if (
            ch in _APOSTROPHES
            and 0 < i < last
            and _wordlike(text[i - 1])
            and _wordlike(text[i + 1])
        ):
            continue
        category = unicodedata.category(ch)
        if ch in _JOINERS or ch in _KEPT_PUNCTUATION or category[0] in "LMNS":
            out.append(ch)
        elif category == "Cf":
            continue  # other invisible format characters (soft hyphen, bidi marks)
        else:
            out.append(" ")  # punctuation, separators, control characters
    return " ".join("".join(out).split())


def speaker_labels(speakers: Sequence[str]) -> list[str]:
    """`s<n>` labels numbered by first appearance."""
    numbers: dict[str, int] = {}
    return [f"s{numbers.setdefault(s, len(numbers) + 1)}" for s in speakers]


def clean_window(turns: Sequence[tuple[str, str]]) -> CleanWindow:
    """Clean (speaker, text) turns, oldest first, into the model input."""
    labels = speaker_labels([speaker for speaker, _ in turns])
    texts = []
    for speaker, text in turns:
        cleaned = clean_text(text)
        if not cleaned:
            raise EmptyTurnError(f"turn by {speaker!r} is empty after cleaning")
        texts.append(cleaned)
    transcript = "\n".join(f"{label}: {text}" for label, text in zip(labels, texts, strict=True))
    return CleanWindow(transcript=transcript, speaker_labels=labels, texts=texts)
