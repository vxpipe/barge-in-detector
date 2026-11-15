"""Train / validation / test split: every combination is spread over the splits."""

from collections import Counter, defaultdict

from barge_in_detector.splitting import SPLITS, split_rows


def rows(per_combination, languages=("en", "hi"), labels=("backchannel", "barge_in")):
    out = []
    n = 0
    for language in languages:
        for label in labels:
            for c in range(6):
                cid = f"{language}-{label}-{c}"
                for _ in range(per_combination):
                    n += 1
                    out.append(
                        {
                            "utterance_id": f"u{n:05d}",
                            "combination_id": cid,
                            "language": language,
                            "label": label,
                            "status": "accepted",
                            "text_clean": f"t{n}",
                        }
                    )
    return out


def test_every_combination_has_samples_in_train_and_one_held_out():
    splits = split_rows(rows(5))
    by_combination = defaultdict(Counter)
    for name, items in splits.items():
        for r in items:
            by_combination[r["combination_id"]][name] += 1
    for counts in by_combination.values():
        assert counts["train"] == 4
        assert counts["validation"] + counts["test"] == 1


def test_validation_and_test_cover_every_language_and_label():
    splits = split_rows(rows(5))
    for name in ("validation", "test"):
        assert {(r["language"], r["label"]) for r in splits[name]} == {
            (lang, label) for lang in ("en", "hi") for label in ("backchannel", "barge_in")
        }
    sizes = {name: len(splits[name]) for name in SPLITS}
    assert sizes == {"train": 96, "validation": 12, "test": 12}


def test_a_single_sample_stays_in_training():
    splits = split_rows(rows(1))
    assert len(splits["train"]) == 24
    assert not splits["validation"] and not splits["test"]


def test_only_accepted_samples_and_the_split_is_deterministic():
    data = rows(5)
    data[0] = {**data[0], "status": "retired"}
    data[1] = {**data[1], "status": "rejected"}
    first, second = split_rows(data), split_rows(list(reversed(data)))
    assert first == second
    ids = {r["utterance_id"] for items in first.values() for r in items}
    assert "u00001" not in ids and "u00002" not in ids
