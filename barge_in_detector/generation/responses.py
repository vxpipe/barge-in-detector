"""Validation of the generator's structured responses. Reject, never repair."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from barge_in_detector.catalogue import UNSPACED_SCRIPTS, Catalogue
from barge_in_detector.cleaner import clean_text, words

LENGTH_BOUNDS = {"w1": (1, 1), "w2_3": (2, 3), "w4_8": (4, 8), "w9_plus": (9, 10_000)}


@dataclass(frozen=True)
class Problem:
    index: int
    reason: str
    label: str | None = None


def _unspaced(cat: Catalogue, spec: Mapping[str, str]) -> bool:
    scripts = set(cat.languages[spec["language"]]["scripts"])
    return spec["script_form"] != "romanized" and bool(scripts & UNSPACED_SCRIPTS)


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

    @property
    def barge_in_from_word(self) -> int | None:
        return self.data["barge_in_from_word"]


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
    """The first problem with one utterance, or None. Lengths and
    barge_in_from_word count cleaner.words."""
    for key, kind in _UTTERANCE_KEYS.items():
        if not isinstance(s.get(key), kind):
            return "malformed_sample"
    if "barge_in_from_word" not in s:
        return "malformed_sample"
    word = s["barge_in_from_word"]
    if word is not None and (not isinstance(word, int) or isinstance(word, bool)):
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
    tokens = words(text)
    if not _unspaced(cat, spec):
        low, high = LENGTH_BOUNDS[spec["length_bucket"]]
        if not low <= len(tokens) <= high:
            return "length_bucket_mismatch"
    if s["label"] == "barge_in":
        if word is None:
            return "barge_in_word_missing"
        if not 0 <= word < max(len(tokens), 1):
            return "barge_in_word_out_of_range"
        if (word == 0) != (spec["development"] == "stable"):
            return "development_word_mismatch"
    elif word is not None:
        return "barge_in_word_not_allowed"
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
