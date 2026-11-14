"""Validation of Gemini's structured responses. Reject, never repair."""

import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from barge_in_detector.catalogue import UNSPACED_SCRIPTS, Catalogue
from barge_in_detector.cleaner import clean_text, words

LENGTH_BOUNDS = {"w1": (1, 1), "w2_3": (2, 3), "w4_8": (4, 8), "w9_plus": (9, 10_000)}
SENTENCE_END = set(".?!।॥。？！؟۔")
CUT_MARKS = ("…", "...", "-", "—", "–", "‒")
_SAMPLE_KEYS = {
    "set_index": int,
    "member_index": int,
    "context_turn": dict,
    "speaker": str,
    "speaker_full_text": str,
    "speaker_spoken_text": str,
    "listener": str,
    "listener_text": str,
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


def _unspaced(cat: Catalogue, spec: Mapping[str, str]) -> bool:
    scripts = set(cat.languages[spec["language"]]["scripts"])
    return spec["script_form"] != "romanized" and bool(scripts & UNSPACED_SCRIPTS)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _same(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def _cut_matches(cut: str, spoken: str, full: str) -> bool:
    """Whether the spoken share of the speaker's turn fits the cut position.

    start: at most 4 words said; early: 8-45% of the words; middle: 30-75%;
    near_end: at least 60% with at most 6 words still to come. The bands
    overlap so natural cuts are not rejected at the edges.
    """
    said, planned = len(words(spoken)), len(words(full))
    share = said / planned
    if cut == "start":
        return said <= 4
    if cut == "early":
        return 0.08 <= share <= 0.45
    if cut == "middle":
        return 0.30 <= share <= 0.75
    if cut == "near_end":
        return share >= 0.60 and planned - said <= 6
    return True


def _check_sample(cat: Catalogue, spec: Mapping[str, str], s: Mapping[str, Any]) -> str | None:
    """The first problem with one sample, or None."""
    for key, kind in _SAMPLE_KEYS.items():
        if not isinstance(s.get(key), kind):
            return "malformed_sample"
    context = s["context_turn"]
    if not all(isinstance(context.get(k), str) for k in ("speaker", "text")):
        return "malformed_sample"
    word = s.get("barge_in_from_word")
    if word is not None and not isinstance(word, int):
        return "malformed_sample"
    if s["label"] not in cat.labels:
        return "unknown_label"

    for text in (
        context["text"],
        s["speaker_spoken_text"],
        s["speaker_full_text"],
        s["listener_text"],
    ):
        if not clean_text(text):
            return "empty_after_cleaning"

    # Speech-to-text output never contains ellipses, in a cut or anywhere else.
    if any(
        "…" in x or "..." in x
        for x in (context["text"], s["speaker_full_text"], s["listener_text"])
    ):
        return "ellipsis_used"

    full, spoken = _norm(s["speaker_full_text"]), _norm(s["speaker_spoken_text"])
    unspaced = _unspaced(cat, spec)
    if not full.startswith(spoken):
        return "spoken_not_prefix"
    if (
        not unspaced
        and len(spoken) < len(full)
        and unicodedata.category(full[len(spoken)])[0] in "LMN"
    ):
        return "spoken_not_at_word_boundary"
    finished = clean_text(spoken) == clean_text(full)
    if finished != (spec["cut_position"] == "just_finished"):
        return "cut_position_mismatch"
    if not finished and spoken.rstrip().endswith(CUT_MARKS):
        # Transcripts just stop where the speaker was cut off: no ellipsis or dash.
        return "cut_marked"
    if not finished and spoken.rstrip()[-1] in SENTENCE_END:
        return "cut_at_sentence_end"
    if not finished and not unspaced and not _cut_matches(spec["cut_position"], spoken, full):
        return "cut_position_mismatch"

    speaker, listener = s["speaker"], s["listener"]
    if _same(speaker, listener) or _same(context["speaker"], speaker):
        return "speaker_pattern_mismatch"
    returning = _same(context["speaker"], listener)
    if returning != (spec["speaker_pattern"] == "returning_speaker"):
        return "speaker_pattern_mismatch"

    candidate_words = words(s["listener_text"])
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
        return (clean_text(s["listener_text"]),)
    if shared == "speaker_turn":
        return (
            clean_text(s["context_turn"]["text"]),
            clean_text(s["speaker_full_text"]),
            clean_text(s["speaker_spoken_text"]),
        )
    if shared == "candidate_opening":
        return tuple(words(s["listener_text"])[:1])
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
    # Soft checks answered no: recorded with their P(yes), never a rejection.
    soft: dict[str, float] = field(default_factory=dict)


_CHECKS = ("language_ok", "script_ok", "cut_ok", "natural_ok")
# Checks that reject a sample: the label is right only if the intent, the cut,
# the language and the script are. Every other question (variety factors such
# as repetition, mood, register, scenario, speaker activity, development, and
# naturalness) is soft: recorded, never a rejection.
HARD_CHECKS = ("language_ok", "script_ok", "cut_ok", "factor_listener_intent")
YES = 0.5


def parse_decision(
    response: Mapping[str, Any], *, hard_checks: Sequence[str] = HARD_CHECKS
) -> Verdict:
    """A verdict from one stored Decisions API response (yes/no answers).

    - label: barge_in when P(is_barge_in) >= 0.5 and P(is_backchannel) < 0.5;
      backchannel for the reverse; ambiguous when both or neither are yes.
    - barge_in_from_word: for a barge-in, the first prefix_<k> answered yes
      gives word k - 1; with none, the full turn decides (its last word).
    - a check answered no (P(yes) < 0.5) rejects the sample only if it is one
      of `hard_checks`; the others are kept in `soft` and the notes. A blocked
      input fails every check.
    """
    if response.get("blocked"):
        return Verdict(-1, "ambiguous", None, [f"blocked:{response['blocked']}"], "input blocked")
    try:
        answers = response["answers"]
        p = {key: float(answer["noul"]) for key, answer in answers.items()}
        back, barge = p["is_backchannel"], p["is_barge_in"]
        checks = {c: p[c] for c in _CHECKS}
        checks |= {k: v for k, v in p.items() if k.startswith("factor_")}
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgementError(f"incomplete decision: {type(exc).__name__}: {exc}") from exc
    if barge >= YES > back:
        label = "barge_in"
    elif back >= YES > barge:
        label = "backchannel"
    else:
        label = "ambiguous"
    word = None
    if label == "barge_in":
        prefixes = sorted(int(k.removeprefix("prefix_")) for k in p if k.startswith("prefix_"))
        word = next((k - 1 for k in prefixes if p[f"prefix_{k}"] >= YES), len(prefixes))
    failed = [c for c in checks if checks[c] < YES and c in hard_checks]
    soft = {c: checks[c] for c in checks if checks[c] < YES and c not in hard_checks}
    notes = f"P(backchannel)={back:.2f}; P(barge_in)={barge:.2f}" + "".join(
        f"; P({c})={checks[c]:.2f}" for c in failed
    )
    if soft:
        notes += "; soft: " + ", ".join(f"P({c})={v:.2f}" for c, v in soft.items())
    return Verdict(-1, label, word, failed, notes, soft)
