"""Validation of Gemini's structured responses. Reject, never repair."""

import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from barge_in_detector.catalogue import UNSPACED_SCRIPTS, Catalogue
from barge_in_detector.cleaner import clean_text

LENGTH_BOUNDS = {"w1": (1, 1), "w2_3": (2, 3), "w4_8": (4, 8), "w9_plus": (9, 10_000)}
SENTENCE_END = set(".?!।॥。？！؟۔")
_SAMPLE_KEYS = {
    "set_index": int,
    "member_index": int,
    "context_turn": dict,
    "agent_speaker": str,
    "agent_full_text": str,
    "agent_spoken_text": str,
    "candidate_speaker": str,
    "candidate_role": str,
    "candidate_text": str,
    "label": str,
    "rationale": str,
}


@dataclass(frozen=True)
class Sample:
    index: int
    target_index: int
    set_index: int
    data: dict[str, Any]

    @property
    def label(self) -> str:
        return self.data["label"]

    @property
    def barge_in_from_word(self) -> int | None:
        return self.data["barge_in_from_word"]


@dataclass(frozen=True)
class Problem:
    index: int
    target_index: int | None
    reason: str
    label: str | None = None


@dataclass
class ParsedGeneration:
    samples: list[Sample] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    fatal: str | None = None


def words(text: str) -> list[str]:
    return clean_text(text).split()


def _unspaced(cat: Catalogue, spec: Mapping[str, str]) -> bool:
    scripts = set(cat.languages[spec["language"]]["scripts"])
    return spec["script_form"] != "romanized" and bool(scripts & UNSPACED_SCRIPTS)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _same(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def _check_sample(cat: Catalogue, spec: Mapping[str, str], s: Mapping[str, Any]) -> str | None:
    """The first problem with one sample, or None."""
    for key, kind in _SAMPLE_KEYS.items():
        if not isinstance(s.get(key), kind):
            return "malformed_sample"
    context = s["context_turn"]
    if not all(isinstance(context.get(k), str) for k in ("speaker", "role", "text")):
        return "malformed_sample"
    word = s.get("barge_in_from_word")
    if word is not None and not isinstance(word, int):
        return "malformed_sample"
    if s["label"] not in cat.labels:
        return "unknown_label"

    for text in (
        context["text"],
        s["agent_spoken_text"],
        s["agent_full_text"],
        s["candidate_text"],
    ):
        if not clean_text(text):
            return "empty_after_cleaning"

    full, spoken = _norm(s["agent_full_text"]), _norm(s["agent_spoken_text"])
    unspaced = _unspaced(cat, spec)
    if not full.startswith(spoken):
        return "agent_spoken_not_prefix"
    if (
        not unspaced
        and len(spoken) < len(full)
        and unicodedata.category(full[len(spoken)])[0] in "LMN"
    ):
        return "agent_spoken_not_at_word_boundary"
    finished = clean_text(spoken) == clean_text(full)
    if finished != (spec["cut_position"] == "just_finished"):
        return "cut_position_mismatch"
    if not finished and spoken.rstrip()[-1] in SENTENCE_END:
        return "agent_cut_at_sentence_end"

    agent, candidate = s["agent_speaker"], s["candidate_speaker"]
    if _same(agent, candidate) or context["role"] == "agent" or _same(context["speaker"], agent):
        return "speaker_pattern_mismatch"
    returning = _same(context["speaker"], candidate)
    if returning != (spec["speaker_pattern"] == "returning_speaker"):
        return "speaker_pattern_mismatch"

    candidate_words = words(s["candidate_text"])
    if not unspaced:
        low, high = LENGTH_BOUNDS[spec["length_bucket"]]
        if not low <= len(candidate_words) <= high:
            return "length_bucket_mismatch"

    if s["label"] == "barge_in":
        if word is None:
            return "barge_in_word_missing"
        if not 0 <= word < max(len(candidate_words), 1):
            return "barge_in_word_out_of_range"
        if (word == 0) != (spec["development"] == "stable"):
            return "development_word_mismatch"
    elif word is not None:
        return "barge_in_word_not_allowed"
    return None


def _shared_key(group: Mapping[str, Any], s: Mapping[str, Any]) -> tuple[str, ...]:
    shared = group["shared"]
    if shared == "candidate_text":
        return (clean_text(s["candidate_text"]),)
    if shared == "agent_turn":
        return (
            clean_text(s["context_turn"]["text"]),
            clean_text(s["agent_full_text"]),
            clean_text(s["agent_spoken_text"]),
        )
    if shared == "candidate_opening":
        return tuple(words(s["candidate_text"])[:1])
    raise ValueError(f"unknown shared element {shared}")


def parse_generation(
    cat: Catalogue,
    text: str,
    *,
    specs: Sequence[Mapping[str, str]],
    count: int,
    group: Mapping[str, Any] | None = None,
) -> ParsedGeneration:
    """Validate a generator response.

    `count` is the number of samples requested (sets × members for a group).
    """
    result = ParsedGeneration()
    try:
        data = json.loads(text)
        raw = data["samples"]
        if not isinstance(raw, list):
            raise TypeError("samples is not a list")
    except (ValueError, KeyError, TypeError) as exc:
        result.fatal = f"unparseable response: {type(exc).__name__}"
        return result

    members = len(specs)
    sets = count // members
    by_set: dict[int, dict[int, Sample]] = {}
    for index, s in enumerate(raw):
        if not isinstance(s, dict):
            result.problems.append(Problem(index, None, "malformed_sample"))
            continue
        set_index = s.get("set_index", 0) if group else 0
        member = s.get("member_index", 0) if group else 0
        if group is None:
            set_index = index
        label = s.get("label") if isinstance(s.get("label"), str) else None
        if (
            not isinstance(set_index, int)
            or not isinstance(member, int)
            or not 0 <= member < members
        ):
            result.problems.append(Problem(index, None, "malformed_sample", label))
            continue
        if set_index >= sets or member in by_set.get(set_index, {}):
            result.problems.append(Problem(index, member, "extra_sample", label))
            continue
        reason = _check_sample(cat, specs[member], s)
        if reason:
            result.problems.append(Problem(index, member, reason, label))
            by_set.setdefault(set_index, {})[member] = None  # type: ignore[assignment]
            continue
        by_set.setdefault(set_index, {})[member] = Sample(index, member, set_index, s)

    for set_index in sorted(by_set):
        entries = by_set[set_index]
        good = [x for x in entries.values() if x is not None]
        if group is None:
            result.samples += good
            continue
        if len(good) < members:
            reason = (
                "contrastive_member_rejected"
                if len(entries) == members
                else "contrastive_set_incomplete"
            )
            result.problems += [Problem(x.index, x.target_index, reason, x.label) for x in good]
            continue
        if len({_shared_key(group, x.data) for x in good}) > 1:
            result.problems += [
                Problem(x.index, x.target_index, "contrastive_shared_mismatch", x.label)
                for x in good
            ]
            continue
        result.samples += sorted(good, key=lambda x: x.target_index)
    return result


# ---------------------------------------------------------------- judge


class JudgementError(ValueError):
    """The judge response is unusable."""


@dataclass(frozen=True)
class Verdict:
    index: int
    label: str
    barge_in_from_word: int | None
    checks_failed: list[str]
    notes: str


_CHECKS = ("language_ok", "script_ok", "cut_ok", "natural_ok")


def parse_judgement(text: str, *, count: int) -> dict[int, Verdict]:
    try:
        items = json.loads(text)["items"]
        verdicts = {}
        for item in items:
            index = item["index"]
            if not isinstance(index, int) or item["label"] not in (
                "backchannel",
                "barge_in",
                "ambiguous",
            ):
                raise TypeError("bad item")
            word = item.get("barge_in_from_word")
            if word is not None and not isinstance(word, int):
                raise TypeError("bad barge_in_from_word")
            verdicts[index] = Verdict(
                index=index,
                label=item["label"],
                barge_in_from_word=word,
                checks_failed=[c for c in _CHECKS if item.get(c) is not True],
                notes=str(item.get("notes", "")),
            )
    except (ValueError, KeyError, TypeError) as exc:
        raise JudgementError(f"unparseable judgement: {type(exc).__name__}") from exc
    if set(verdicts) != set(range(count)):
        raise JudgementError(f"judgement covers {sorted(verdicts)}, expected 0..{count - 1}")
    return verdicts
