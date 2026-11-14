"""Example builder, version 1: conversations to model inputs.

One example per (turn shape x candidate prefix). Prefixes are cut at the
original candidate's whitespace tokens, then each window is cleaned; a token
that adds no cleaned word (a lone dash) yields no new prefix. The prefix label
follows the conversation's label: for a barge-in, prefixes before
barge_in_from_word are backchannel. Words are counted as cleaner.words does
("Mm-hmm," is one word), the same way the generator and validator count them.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from barge_in_detector import ids
from barge_in_detector.cleaner import (
    CLEANER_VERSION,
    clean_text,
    clean_window,
    raw_window,
    words,
)
from barge_in_detector.storage.schema import FACTOR_COLUMNS

BUILDER_VERSION = "1"
LABEL_IDS = {"backchannel": 0, "barge_in": 1, "ambiguous": None}
_FACTORS = [c for c, _ in FACTOR_COLUMNS if c != "label"]


def _prefix_label(conversation: Mapping[str, Any], word_count: int) -> str:
    label = conversation["label"]
    if label != "barge_in":
        return label
    return "barge_in" if word_count > conversation["barge_in_from_word"] else "backchannel"


def build_examples(conversations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in conversations:
        context = c["turns"][-1]
        agent = {"speaker": c["agent_speaker"], "role": "agent", "text": c["agent_spoken_text"]}
        tokens = c["candidate_text"].split()
        prefixes: list[tuple[int, str]] = []
        last_clean = ""
        for index in range(len(tokens)):
            text = " ".join(tokens[: index + 1])
            cleaned = clean_text(text)
            if cleaned and cleaned != last_clean:
                prefixes.append((index, text))
                last_clean = cleaned
        for shape in (3, 2):
            for position, (index, text) in enumerate(prefixes):
                candidate = {
                    "speaker": c["candidate_speaker"],
                    "role": c["candidate_role"],
                    "text": text,
                }
                turns = ([context] if shape == 3 else []) + [agent, candidate]
                window = clean_window([(t["speaker"], t["text"]) for t in turns])
                candidate_clean = window.texts[-1]
                label = _prefix_label(c, len(words(text)))
                rows.append(
                    {
                        "example_id": ids.example_id(
                            c["conversation_id"], turn_shape=shape, prefix_index=index
                        ),
                        **{f: c[f] for f in _FACTORS},
                        "label": label,
                        "conversation_id": c["conversation_id"],
                        "run_id": c["run_id"],
                        "builder_version": BUILDER_VERSION,
                        "turn_shape": shape,
                        "turns_original": [dict(t) for t in turns],
                        "transcript_original": "\n".join(
                            f"{t['speaker']}: {t['text']}" for t in turns
                        ),
                        "transcript_clean": window.transcript,
                        "transcript_raw": raw_window(
                            [(t["speaker"], t["text"]) for t in turns]
                        ).transcript,
                        "speaker_labels": window.speaker_labels,
                        "candidate_original": text,
                        "candidate_clean": candidate_clean,
                        "cleaner_version": CLEANER_VERSION,
                        "prefix_index": index,
                        "is_final_prefix": position == len(prefixes) - 1,
                        "label_id": LABEL_IDS[label],
                    }
                )
    rows.sort(key=lambda r: r["example_id"])
    return rows
