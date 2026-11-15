import pytest

from barge_in_detector.splitting import SPLITS
from barge_in_detector.supplements import additive_splits


def row(id, text=None, cid="new", label="backchannel"):
    return {
        "utterance_id": id,
        "combination_id": cid,
        "text_clean": text or id,
        "language": "en",
        "label": label,
        "status": "accepted",
    }


def baseline():
    return {s: [row("old-" + s, cid="old")] for s in SPLITS}


def assignments(splits):
    return {r["utterance_id"]: s for s, rows in splits.items() for r in rows}


def test_additive_splits_preserve_original_and_previous_supplement_assignments():
    original = baseline()
    first, combined = additive_splits(original, [row("a"), row("b"), row("c")], reserved=set())
    for s in SPLITS:
        assert combined[s][: len(original[s])] == original[s]
    grown, _ = additive_splits(original, [row(x) for x in "abcdef"], reserved=set(), previous=first)
    assert all(assignments(grown)[id] == s for id, s in assignments(first).items())
    assert sum(map(len, grown.values())) == 6
    assert any(grown[s] for s in ("validation", "test"))


def test_singleton_can_gain_a_heldout_sample_without_moving_its_training_row():
    first, _ = additive_splits(baseline(), [row("a")], reserved=set())
    grown, _ = additive_splits(baseline(), [row("a"), row("b")], reserved=set(), previous=first)
    assert assignments(grown)["a"] == "train"
    assert assignments(grown)["b"] != "train"


@pytest.mark.parametrize("text", ["old-test", "reviewed", "same"])
def test_duplicate_or_reviewed_texts_are_refused(text):
    rows = [row("a", text)]
    if text == "same":
        rows.append(row("b", text))
    with pytest.raises(ValueError, match="duplicate|reserved"):
        additive_splits(baseline(), rows, reserved={"reviewed"})
