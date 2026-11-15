"""The working store: one utterances table (exported as one CSV) and a local request log."""

import json
from pathlib import Path

import duckdb
import pytest

from barge_in_detector.storage.schema import SCHEMA_VERSION
from barge_in_detector.storage.store import RefusedUpdate, Store

CID = "0198f0c4-0000-5000-8000-000000000001"
TARGET = {
    "combination_id": CID,
    "language": "hi",
    "regional_variant": "hi",
    "script_form": "romanized",
    "code_switching": "within_sentence",
    "code_switch_set": "en",
    "mixed_in_script": "default",
    "caller_intent": "correction",
    "development": "backchannel_then_barge_in",
    "repetition": "repeated",
    "length_bucket": "w4_8",
    "locales": ["hi-Latn-IN", "en-IN"],
    "split": "train",
    "label": "barge_in",
}


def row(text, status="accepted", **extra):
    return {
        "status": status,
        "reason": None if status == "accepted" else "judge script_hi=0.1",
        "text": text,
        "label": "barge_in",
        "raw_sample": {"text": text, "label": "barge_in"},
        "judge_answers": {"script_hi": {"noul": 0.9}} if status == "accepted" else None,
        "text_hash": f"h-{text}",
        "request_id": None,
        "judge_request_id": None,
        "sample_index": 0,
        "model_id": "m",
        "prompt_template": "p",
        "judge_model_id": "j",
        "judge_template": "t",
        **extra,
    }


@pytest.fixture
def store(tmp_path: Path):
    s = Store.open(tmp_path / "work.duckdb")
    yield s
    s.close()


def test_accepted_and_rejected_samples_are_rows_of_one_table(store):
    store.insert_utterance(TARGET, row("Haan haan, but change it."))
    store.insert_utterance(TARGET, row("Haan haan, wait.", status="rejected"))
    assert store.accepted_counts() == {store.utterances()[0]["combination_id"]: 1}
    assert store.accepted_text_hashes() == {"h-Haan haan, but change it."}
    rejected = store.utterances(status="rejected")[0]
    assert rejected["reason"].startswith("judge")
    assert rejected["text_clean"] == "haan haan wait"


def test_export_writes_one_csv_of_utterances_only(store, tmp_path):
    store.insert_utterance(TARGET, row("Haan haan, but change it."))
    store.insert_request(
        kind="generate_utterances",
        combination_id=CID,
        language="hi",
        requested_count=1,
        model_id="m",
        prompt_template="p",
        prompt_hash="h",
    )
    out = tmp_path / "export"
    store.export(out)
    assert sorted(p.name for p in out.iterdir()) == ["utterances.csv"]


def test_export_round_trips_and_is_stable(store, tmp_path):
    awkward = 'He said "ok, ok",\nthen: हाँ, wait'
    store.insert_utterance(TARGET, row(awkward))
    store.insert_utterance(TARGET, row("Hmm.", status="rejected", judge_answers=None))
    out = tmp_path / "export"
    assert store.export(out) == [out / "utterances.csv"]
    before = (out / "utterances.csv").stat().st_mtime_ns
    assert store.export(out) == []  # unchanged data leaves the file untouched
    assert (out / "utterances.csv").stat().st_mtime_ns == before

    rebuilt = Store.open(tmp_path / "rebuilt.duckdb")
    try:
        rebuilt.import_exports(out)
        original, copy = store.utterances(status=None), rebuilt.utterances(status=None)
        assert copy == original
        assert copy[0]["text"] == awkward
        assert copy[0]["locales"] == ["hi-Latn-IN", "en-IN"]
        assert json.loads(copy[0]["judge_answers"]) == {"script_hi": {"noul": 0.9}}
        assert copy[1]["judge_answers"] is None
        assert "barge_in_from_word" not in copy[0]
    finally:
        rebuilt.close()


def test_retiring_keeps_the_row(store):
    utterance_id = store.insert_utterance(TARGET, row("Haan haan, but change it."))
    store.retire_utterance(utterance_id, "bad prompt version")
    assert store.accepted_counts() == {}
    assert store.utterances(status="retired")[0]["reason"] == "bad prompt version"
    with pytest.raises(RefusedUpdate):
        store.retire_utterance(utterance_id, "again")


def test_request_log_tracks_status_and_cost(store):
    request_id = store.insert_request(
        kind="generate_utterances",
        combination_id=CID,
        language="hi",
        requested_count=3,
        model_id="m",
        prompt_template="p",
        prompt_hash="h",
    )
    store.record_response(request_id, response={"text": "{}"}, usage={}, cost_usd=0.01)
    with pytest.raises(RefusedUpdate):
        store.record_response(request_id, response={}, usage={}, cost_usd=0.0)
    store.mark_processed(request_id)
    assert store.request(request_id)["status"] == "processed"
    assert store.requested_counts("generate_utterances", prompt_template="p", model_id="m")
    assert store.total_cost() == pytest.approx(0.01)


def test_an_older_database_is_refused_with_a_clear_message(tmp_path):
    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
    con.execute("INSERT INTO meta VALUES ('schema_version', '5')")
    con.close()
    with pytest.raises(RuntimeError, match="schema 5.*delete"):
        Store.open(path)
    assert SCHEMA_VERSION == "7"
