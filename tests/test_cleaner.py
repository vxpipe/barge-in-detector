from pathlib import Path

import pytest
import yaml

from barge_in_detector import cleaner

GOLDEN = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "cleaner_golden.yaml").read_text(encoding="utf-8")
)


def test_golden_version_matches_cleaner():
    assert GOLDEN["cleaner_version"] == cleaner.CLEANER_VERSION


@pytest.mark.parametrize("case", GOLDEN["texts"], ids=lambda c: c["name"])
def test_clean_text_golden(case):
    assert cleaner.clean_text(case["text"]) == case["clean"]


@pytest.mark.parametrize("case", GOLDEN["windows"], ids=lambda c: c["name"])
def test_clean_window_golden(case):
    turns = [(t["speaker"], t["text"]) for t in case["turns"]]
    assert cleaner.clean_window(turns).transcript == case["clean"]


def test_speaker_labels_by_first_appearance():
    window = cleaner.clean_window([("A", "one"), ("B", "two"), ("A", "three")])
    assert window.speaker_labels == ["s1", "s2", "s1"]


def test_empty_turn_is_rejected():
    with pytest.raises(cleaner.EmptyTurnError):
        cleaner.clean_window([("A", "hello"), ("B", "…!")])


def test_words_are_original_tokens_with_content():
    assert cleaner.words("Mm-hmm, — wait!") == ["Mm-hmm,", "wait!"]


def test_raw_window_keeps_casing_and_punctuation():
    window = cleaner.raw_window(
        [
            ("Hotel Manager", "Sir,  we have a suite on the\u00a0highest floor, and"),
            ("Guest", "OK!"),
        ]
    )
    assert window.transcript == "s1: Sir, we have a suite on the highest floor, and\ns2: OK!"
    assert window.speaker_labels == ["s1", "s2"]
