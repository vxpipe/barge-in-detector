"""Evaluation metrics for the two labels."""

import pytest

from barge_in_detector.training.metrics import compute


def test_metrics_report_false_and_missed_interruptions():
    labels = ["backchannel", "backchannel", "barge_in", "barge_in", "barge_in"]
    preds = ["backchannel", "barge_in", "barge_in", "barge_in", "backchannel"]
    languages = ["en", "en", "en", "hi", "hi"]
    m = compute(labels, preds, languages)
    assert m["accuracy"] == pytest.approx(3 / 5)
    # A backchannel predicted as barge-in cuts the agent off for nothing.
    assert m["false_interruption_rate"] == pytest.approx(1 / 2)
    # A barge-in predicted as backchannel lets the agent talk over the caller.
    assert m["missed_interruption_rate"] == pytest.approx(1 / 3)
    assert m["per_class"]["barge_in"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["barge_in"]["recall"] == pytest.approx(2 / 3)
    assert m["per_language"]["en"]["accuracy"] == pytest.approx(2 / 3)
    assert m["per_language"]["hi"]["count"] == 2
    assert 0 < m["macro_f1"] < 1


def test_perfect_predictions():
    labels = ["backchannel", "barge_in"]
    m = compute(labels, labels, ["en", "en"])
    assert m["accuracy"] == m["macro_f1"] == 1.0
    assert m["false_interruption_rate"] == m["missed_interruption_rate"] == 0.0
