"""Validation of the generator's structured responses. Reject, never repair."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from barge_in_detector.catalogue import Catalogue
from barge_in_detector.cleaner import clean_text


@dataclass(frozen=True)
class Problem:
    index: int
    reason: str
    label: str | None = None


_UTTERANCE_KEYS = {"text": str, "label": str, "rationale": str}
# The utterance track has two labels only; see generation/factors/utterances.yaml.
UTTERANCE_LABELS = ("backchannel", "barge_in")
_TAG_MARKS = ("[", "]", "(", ")", "<", ">")


@dataclass(frozen=True)
class UtteranceSample:
    index: int
    data: dict[str, Any]

    @property
    def label(self) -> str:
        return self.data["label"]


@dataclass
class ParsedUtterances:
    samples: list[UtteranceSample] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    fatal: str | None = None


def _check_utterance(
    cat: Catalogue,
    spec: Mapping[str, str],
    s: Mapping[str, Any],
    labels: Sequence[str] = UTTERANCE_LABELS,
) -> str | None:
    """The first problem with one utterance, or None."""
    for key, kind in _UTTERANCE_KEYS.items():
        if not isinstance(s.get(key), kind):
            return "malformed_sample"
    if s["label"] not in labels:
        return "unknown_label"
    text = s["text"]
    if not clean_text(text):
        return "empty_after_cleaning"
    if "…" in text or "..." in text:
        return "ellipsis_used"
    if any(mark in text for mark in _TAG_MARKS):
        return "tag_used"
    return None


def parse_utterances(
    cat: Catalogue,
    text: str,
    *,
    spec: Mapping[str, str],
    count: int,
    labels: Sequence[str] = UTTERANCE_LABELS,
) -> ParsedUtterances:
    """Validate an utterance-track response. Reject, never repair."""
    result = ParsedUtterances()
    try:
        raw = json.loads(text)["samples"]
        if not isinstance(raw, list):
            raise TypeError("samples is not a list")
    except (ValueError, KeyError, TypeError) as exc:
        result.fatal = f"unparseable response: {type(exc).__name__}"
        return result
    for index, s in enumerate(raw):
        label = s.get("label") if isinstance(s, dict) and isinstance(s.get("label"), str) else None
        if index >= count:
            result.problems.append(Problem(index, "extra_sample", label))
            continue
        reason = (
            "malformed_sample"
            if not isinstance(s, dict)
            else _check_utterance(cat, spec, s, labels)
        )
        if reason:
            result.problems.append(Problem(index, reason, label))
            continue
        result.samples.append(UtteranceSample(index, s))
    return result
