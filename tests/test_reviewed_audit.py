import hashlib
import json

import pytest

from experiments.audit_reviewed_errors import coverage, near_duplicates, reviewed_predictions


def fixture_files(tmp_path, decision="barge_in"):
    source = tmp_path / "challenge.csv"
    source.write_text(
        "case_id,language,category,group,text,text_clean,label\n"
        "one,en,short_interruption,,Wait!,wait,backchannel\n"
    )
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "reviews": {"one": {"decision": decision}},
            }
        )
    )
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("case_id,text_clean,p_barge_in\none,wait,0.01\n")
    return source, reviews, predictions


def test_saved_review_overrides_authored_label(tmp_path):
    rows = reviewed_predictions(*fixture_files(tmp_path))
    assert rows[0]["label"] == "barge_in"
    assert rows[0]["authored_label"] == "backchannel"
    assert rows[0]["p_barge_in"] == 0.01


@pytest.mark.parametrize("decision", ["needs_context", "", None])
def test_requires_complete_binary_review(tmp_path, decision):
    with pytest.raises(ValueError, match="binary review"):
        reviewed_predictions(*fixture_files(tmp_path, decision))


def test_rejects_stale_predictions_and_reviews(tmp_path):
    source, reviews, predictions = fixture_files(tmp_path)
    predictions.write_text("case_id,text_clean,p_barge_in\none,stop,0.01\n")
    with pytest.raises(ValueError, match="text mismatch"):
        reviewed_predictions(source, reviews, predictions)
    source.write_text(source.read_text().replace("Wait!", "WAIT!"))
    with pytest.raises(ValueError, match="different dataset"):
        reviewed_predictions(source, reviews, predictions)


@pytest.mark.parametrize("probability", ["nan", "inf", "-0.1", "1.1"])
def test_rejects_invalid_probability(tmp_path, probability):
    source, reviews, predictions = fixture_files(tmp_path)
    predictions.write_text(f"case_id,text_clean,p_barge_in\none,wait,{probability}\n")
    with pytest.raises(ValueError, match="probability"):
        reviewed_predictions(source, reviews, predictions)


def test_rejects_duplicate_predictions(tmp_path):
    source, reviews, predictions = fixture_files(tmp_path)
    predictions.write_text(predictions.read_text() + "one,wait,0.02\n")
    with pytest.raises(ValueError, match="unique case"):
        reviewed_predictions(source, reviews, predictions)


def test_coverage_groups_lengths_within_language_and_label():
    rows = [
        {
            "language": "en",
            "label": "barge_in",
            "caller_intent": "floor_request",
            "text_clean": text,
        }
        for text in [
            "stop",
            "wait please",
            "please wait a moment",
            "can you please wait just one moment",
        ]
    ]
    groups = coverage(rows, ("language", "label", "caller_intent"))
    assert len(groups) == 1
    assert groups[0]["length_counts"] == {"1-2": 2, "3-5": 1, "6+": 1}


def test_near_duplicates_exclude_exact_and_other_languages():
    train = [
        {
            "language": "en",
            "text_clean": "please change my booking to tuesday",
            "label": "barge_in",
        },
        {
            "language": "hi",
            "text_clean": "please change my booking to wednesday",
            "label": "barge_in",
        },
    ]
    heldout = [
        {
            "language": "en",
            "text_clean": "please change my booking to tuesday",
            "label": "barge_in",
        },
        {
            "language": "en",
            "text_clean": "please change my booking to tuesdays",
            "label": "barge_in",
        },
    ]
    result = near_duplicates(train, heldout, threshold=0.8)
    assert result["matched_rows"] == 1
    assert result["examples"][0]["heldout_text"].endswith("tuesdays")
