"""Utterance track: plan, prompts, validation, storage and the fill-to-count run."""

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from fake_llm import FakeLlm

from barge_in_detector import ids
from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation import prompts
from barge_in_detector.generation.config import GenerationConfig
from barge_in_detector.generation.responses import parse_utterances
from barge_in_detector.generation.utterance_runner import UtteranceRunner
from barge_in_detector.generation.utterances import (
    UtterancePlanConfig,
    build_utterance_plan,
    load_utterance_track,
)
from barge_in_detector.storage.schema import SCHEMA_VERSION
from barge_in_detector.storage.store import Store

CAT = load_catalogue()
TRACK = load_utterance_track(CAT)
PLAN = UtterancePlanConfig(
    name="utterance-test",
    seed=7,
    languages=["en", "hi"],
    criteria_per_language=12,
    required_count=3,
    label_shares={"backchannel": 0.5, "barge_in": 0.5},
)
SETTINGS = GenerationConfig(batch_size=10, max_attempts_per_sample=2, concurrency=1)


def spec_of(target):
    return dict(target.spec)


# ---------------------------------------------------------------- track and plan


def test_track_excludes_intents_that_need_a_finished_turn_or_context():
    assert "answer_to_question" not in TRACK.intents
    assert "tracking_repeat" not in TRACK.intents
    assert {"continuer", "floor_request"} <= set(TRACK.intents)
    assert set(TRACK.factors) <= set(CAT.spec_factors)


def test_plan_is_deterministic():
    a = build_utterance_plan(CAT, TRACK, PLAN)
    b = build_utterance_plan(CAT, TRACK, PLAN)
    assert a == b
    assert len(a) == 24


def test_plan_grows_prefix_stable():
    small = build_utterance_plan(CAT, TRACK, PLAN)
    big = build_utterance_plan(CAT, TRACK, replace(PLAN, criteria_per_language=20))
    by_language = {}
    for t in big:
        by_language.setdefault(t.spec["language"], []).append(t)
    for language in ("en", "hi"):
        small_ids = [t.combination_id for t in small if t.spec["language"] == language]
        assert [t.combination_id for t in by_language[language][:12]] == small_ids


def test_plan_specs_are_valid_unique_and_balanced():
    plan = build_utterance_plan(CAT, TRACK, PLAN)
    assert len({t.combination_id for t in plan}) == len(plan)
    for t in plan:
        assert set(t.spec) == set(TRACK.factors)
        assert not CAT.violations(t.spec, partial_spec=True)
        assert t.spec["caller_intent"] in TRACK.intents
        assert t.intended_label == CAT.intents[t.spec["caller_intent"]]["label"]
        assert t.split in ("train", "validation", "test")
        assert t.required_count == 3
    labels = [t.intended_label for t in plan if t.spec["language"] == "en"]
    assert labels.count("backchannel") == labels.count("barge_in") == 6


def test_plan_labels_are_only_backchannel_or_barge_in():
    assert TRACK.labels == ("backchannel", "barge_in")
    assert {CAT.intents[i]["label"] for i in TRACK.intents} == set(TRACK.labels)
    with pytest.raises(ValueError, match="ambiguous"):
        build_utterance_plan(CAT, TRACK, replace(PLAN, label_shares={"ambiguous": 1.0}))


def test_utterance_ids_never_collide_with_conversation_ids():
    t = build_utterance_plan(CAT, TRACK, PLAN)[0]
    assert t.combination_id == ids.utterance_combination_id(t.spec)
    assert t.combination_id != ids.combination_id(t.spec)


# ---------------------------------------------------------------- prompts


def test_system_prompt_lists_only_track_intents():
    system = prompts.render_utterance_system(CAT, TRACK)
    assert "`continuer`" in system
    assert "`answer_to_question`" not in system
    assert "$" not in system  # every placeholder substituted


def test_request_names_every_factor():
    t = build_utterance_plan(CAT, TRACK, PLAN)[0]
    user = prompts.render_utterance_request(CAT, t.spec, count=4)
    assert user.startswith("Write 4 utterances")
    for prefix in ("Language:", "Script:", "Intent:", "Development:", "Repetition:", "Length:"):
        assert f"- {prefix}" in user
    assert f"(intended label: {t.intended_label})" in user


# ---------------------------------------------------------------- validation

EN = {
    "language": "en",
    "regional_variant": "en-US",
    "script_form": "native",
    "code_switching": "none",
    "code_switch_set": "none",
    "mixed_in_script": "default",
    "caller_intent": "floor_request",
    "development": "backchannel_then_barge_in",
    "repetition": "single",
    "length_bucket": "w4_8",
}


def sample(text="Yeah, wait, hold on.", label="barge_in", word=1):
    return {"text": text, "label": label, "barge_in_from_word": word, "rationale": "r"}


def parse(*samples, count=None, spec=EN):
    text = json.dumps({"samples": list(samples)})
    return parse_utterances(CAT, text, spec=spec, count=count or len(samples))


def test_valid_utterance_passes():
    parsed = parse(sample())
    assert parsed.fatal is None and not parsed.problems
    assert parsed.samples[0].data["text"] == "Yeah, wait, hold on."


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (sample("Wait."), "length_bucket_mismatch"),
        (sample("Yeah, wait... hold on."), "ellipsis_used"),
        (sample("Yeah [laughs] wait hold on."), "tag_used"),
        (sample(word=0), "development_word_mismatch"),
        (sample(label="backchannel", word=1), "barge_in_word_not_allowed"),
        (sample(label="maybe"), "unknown_label"),
        (sample(label="ambiguous", word=None), "unknown_label"),
        ({"text": "x"}, "malformed_sample"),
    ],
)
def test_bad_utterances_are_rejected(bad, reason):
    parsed = parse(bad)
    assert [p.reason for p in parsed.problems] == [reason]
    assert not parsed.samples


def test_extra_samples_and_unparseable_responses():
    parsed = parse(sample(), sample("Umm, wait, one second."), count=1)
    assert [p.reason for p in parsed.problems] == ["extra_sample"]
    bad = parse_utterances(CAT, "not json", spec=EN, count=1)
    assert bad.fatal and bad.fatal.startswith("unparseable")


# ---------------------------------------------------------------- storage and runs


@pytest.fixture
def store(tmp_path: Path):
    s = Store.open(tmp_path / "work.duckdb")
    s.apply_utterance_plan(CAT, TRACK, PLAN, build_utterance_plan(CAT, TRACK, PLAN))
    yield s
    s.close()


def run(store, client, **kwargs):
    return UtteranceRunner(store, CAT, TRACK, client, SETTINGS).run(**kwargs)


def filled(store):
    counts = store.utterance_accepted_counts()
    return [counts.get(t["combination_id"], 0) for t in store.utterance_targets()]


def files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.parquet"))}


def test_applying_the_same_plan_twice_changes_nothing(store):
    plan = build_utterance_plan(CAT, TRACK, PLAN)
    assert store.apply_utterance_plan(CAT, TRACK, PLAN, plan) is None
    raised = [replace(t, required_count=5) for t in plan]
    assert store.apply_utterance_plan(CAT, TRACK, PLAN, raised) is not None
    assert {t["required_count"] for t in store.utterance_targets()} == {5}


def test_run_fills_every_criterion_without_a_judge(store):
    client = FakeLlm()
    report = run(store, client, languages=["en"])
    en = [t for t in store.utterance_targets() if t["language"] == "en"]
    counts = store.utterance_accepted_counts()
    assert {counts.get(t["combination_id"], 0) for t in en} == {3}
    assert client.calls == {"generate": 12, "judge": 0}
    assert report.accepted == 36
    row = store.utterances()[0]
    assert row["text_clean"] == row["text"].lower().replace(",", "").replace(".", "")
    assert row["label"] == row["intended_label"]
    assert row["judge_model_id"] is None


def test_second_run_makes_no_calls_and_keeps_the_export(store, tmp_path):
    export = tmp_path / "export"
    run(store, FakeLlm(), export_dir=export)
    before = files(export)
    assert any(p.startswith("utterances/") for p in before)
    client = FakeLlm()
    run(store, client, export_dir=export)
    assert client.calls["generate"] == 0
    assert files(export) == before


def test_limited_run_then_another_fills_exactly_the_rest(store):
    run(store, FakeLlm(), limit=5)
    assert sum(1 for n in filled(store) if n == 3) == 5
    client = FakeLlm()
    run(store, client)
    assert client.calls["generate"] == 19
    assert set(filled(store)) == {3}


def test_label_mismatch_is_rejected_and_the_budget_stops_retries(store):
    client = FakeLlm(label_override="ambiguous")
    report = run(store, client, languages=["en"])
    assert report.accepted == 0
    assert report.reasons["schema:unknown_label"] == 72
    # max_attempts_per_sample 2 x required 3 = 6 samples per criterion: 2 requests.
    assert client.calls["generate"] == 24
    assert len(report.exhausted) == 12


def test_duplicates_of_accepted_text_are_rejected(store):
    report = run(store, FakeLlm(fixed_text="Mm-hmm"), languages=["en"], limit=1)
    assert report.accepted == 1
    assert report.reasons["dedupe:duplicate"] == 2


def test_received_response_is_resumed_before_new_requests(store):
    target = store.utterance_targets()[0]
    run_id = store.start_run(
        name="crashed", revision_id=None, settings={}, generator_model_id="m", judge_model_id=None
    )
    user = prompts.render_utterance_request(CAT, {f: target[f] for f in TRACK.factors}, count=3)
    request_id = store.insert_request(
        run_id=run_id,
        kind=prompts.UTTERANCE_REQUEST_KIND,
        combination_id=target["combination_id"],
        contrastive_group_id=None,
        language=target["language"],
        requested_count=3,
        model_id="m",
        prompt_template=prompts.UTTERANCE_TEMPLATE,
        prompt_hash="h",
    )
    result = FakeLlm().generate(model="m", system_prompt="s", user_prompt=user, schema={})
    store.record_response(request_id, response=result.response, usage=result.usage, cost_usd=0.0)
    client = FakeLlm()
    run(store, client)
    assert store.request(request_id)["status"] == "processed"
    assert client.calls["generate"] == 23  # the resumed criterion needs no new request
    assert set(filled(store)) == {3}


def test_export_then_import_rebuilds_utterances(store, tmp_path):
    run(store, FakeLlm(), languages=["en"], limit=2)
    store.export(tmp_path / "export")
    rebuilt = Store.open(tmp_path / "rebuilt.duckdb")
    try:
        rebuilt.import_exports(tmp_path / "export")
        assert rebuilt.count("utterances") == store.count("utterances") == 6
        assert rebuilt.count("utterance_targets") == 24
    finally:
        rebuilt.close()


def test_schema_3_database_is_migrated(tmp_path):
    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
    con.execute("INSERT INTO meta VALUES ('schema_version', '3')")
    con.close()
    s = Store.open(path)
    try:
        version = s.con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        assert version[0] == SCHEMA_VERSION == "4"
        assert s.count("utterances") == 0
    finally:
        s.close()


def test_code_switching_fits_inside_the_utterance():
    plan = build_utterance_plan(CAT, TRACK, replace(PLAN, criteria_per_language=200))
    for t in plan:
        if t.spec["code_switching"] == "within_sentence":
            assert t.spec["length_bucket"] != "w1"
        if t.spec["code_switching"] == "between_sentences":
            assert t.spec["length_bucket"] in ("w4_8", "w9_plus")


def test_replanning_retires_unused_criteria_the_plan_dropped(store):
    smaller = build_utterance_plan(CAT, TRACK, replace(PLAN, criteria_per_language=10))
    assert store.apply_utterance_plan(CAT, TRACK, PLAN, smaller, retire_missing=True)
    assert len(store.utterance_targets()) == 20
    retired = store.utterance_targets(status="retired")
    assert len(retired) == 4
    assert all(r["retired_reason"].startswith("not in plan revision") for r in retired)


# ---------------------------------------------------------------- parallel requests


class SlowLlm(FakeLlm):
    """Counts how many generate calls overlap."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active = 0
        self.max_active = 0
        self._gate = threading.Lock()

    def generate(self, **kwargs):
        with self._gate:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return super().generate(**kwargs)
        finally:
            with self._gate:
                self.active -= 1


def run_parallel(store, client, concurrency=4, **kwargs):
    settings = replace(SETTINGS, concurrency=concurrency)
    return UtteranceRunner(store, CAT, TRACK, client, settings).run(**kwargs)


def test_parallel_run_overlaps_up_to_concurrency_and_fills_exactly(store):
    client = SlowLlm()
    report = run_parallel(store, client, concurrency=4, languages=["en"])
    assert client.max_active == 4
    assert client.calls["generate"] == 12
    assert report.requests == 12
    assert report.accepted == 36
    en = [t for t in store.utterance_targets() if t["language"] == "en"]
    counts = store.utterance_accepted_counts()
    assert {counts.get(t["combination_id"], 0) for t in en} == {3}


def test_parallel_run_dedupes_across_concurrent_responses(store, tmp_path):
    sequential = Store.open(tmp_path / "sequential.duckdb")
    sequential.apply_utterance_plan(CAT, TRACK, PLAN, build_utterance_plan(CAT, TRACK, PLAN))
    expected = run(sequential, FakeLlm(fixed_text="Mm-hmm"), languages=["en"], limit=8)
    sequential.close()
    report = run_parallel(store, SlowLlm(fixed_text="Mm-hmm"), languages=["en"], limit=8)
    assert report.reasons["dedupe:duplicate"] > 0
    assert report.accepted == expected.accepted
    assert report.reasons["dedupe:duplicate"] == expected.reasons["dedupe:duplicate"]
    texts = [u["text_clean"] for u in store.utterances()]
    assert len(texts) == len(set(texts))


def test_parallel_run_honours_the_limit_and_retries_rejections(store):
    client = SlowLlm()
    run_parallel(store, client, limit=5)
    assert client.calls["generate"] == 5
    rejecting = SlowLlm(label_override="ambiguous")
    report = run_parallel(store, rejecting, languages=["en"])
    # Seven unfilled English criteria, two requests each before the budget stops them.
    assert rejecting.calls["generate"] == 2 * sum(
        1
        for t in store.utterance_targets()
        if t["language"] == "en"
        and store.utterance_accepted_counts().get(t["combination_id"], 0) == 0
    )
    assert report.accepted == 0
