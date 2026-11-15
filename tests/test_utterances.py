"""Utterance track: plan, prompts, validation, storage and the fill-to-count run."""

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

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
    _Builder,
    build_utterance_plan,
    load_utterance_track,
    plan_targets,
)
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
    assert "turn_continuation" not in TRACK.intents  # needs an earlier turn
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
        assert t.required_count == 3
    labels = [t.intended_label for t in plan if t.spec["language"] == "en"]
    assert labels.count("backchannel") == labels.count("barge_in") == 6


def test_plan_labels_are_only_backchannel_or_barge_in():
    assert TRACK.labels == ("backchannel", "barge_in")
    assert {CAT.intents[i]["label"] for i in TRACK.intents} == set(TRACK.labels)
    with pytest.raises(ValueError, match="ambiguous"):
        build_utterance_plan(CAT, TRACK, replace(PLAN, label_shares={"ambiguous": 1.0}))


def test_criterion_id_comes_from_its_factor_values():
    t = build_utterance_plan(CAT, TRACK, PLAN)[0]
    assert t.combination_id == ids.utterance_combination_id(t.spec)


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
    for prefix in ("Languages and scripts", "Intent:"):
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
}


def sample(text="Yeah, wait, hold on.", label="barge_in"):
    return {"text": text, "label": label, "rationale": "r"}


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
        (sample("Yeah, wait... hold on."), "ellipsis_used"),
        (sample("Yeah [laughs] wait hold on."), "tag_used"),
        (sample(label="maybe"), "unknown_label"),
        (sample(label="ambiguous"), "unknown_label"),
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


# ---------------------------------------------------------------- runs

TARGETS = plan_targets(CAT, build_utterance_plan(CAT, TRACK, PLAN))
EN_TARGETS = [t for t in TARGETS if t["language"] == "en"]


@pytest.fixture
def store(tmp_path: Path):
    s = Store.open(tmp_path / "work.duckdb")
    yield s
    s.close()


def run(store, client, settings=SETTINGS, **kwargs):
    return UtteranceRunner(store, CAT, TRACK, client, settings, TARGETS).run(**kwargs)


def filled(store, targets=TARGETS):
    counts = store.accepted_counts()
    return [counts.get(t["combination_id"], 0) for t in targets]


def rejected(store):
    return store.utterances(status="rejected")


def test_plan_targets_carry_what_a_row_needs():
    t = TARGETS[0]
    assert t["label"] in ("backchannel", "barge_in") and t["required_count"] == 3
    assert t["locales"]


def test_run_fills_every_criterion_and_judges_each_utterance(store):
    client = FakeLlm()
    report = run(store, client, languages=["en"])
    assert set(filled(store, EN_TARGETS)) == {3}
    assert client.calls == {"generate": 12, "judge": 36}
    assert report.accepted == 36
    row = store.utterances()[0]
    assert row["text_clean"] == row["text"].lower().replace(",", "").replace(".", "")
    assert row["label"] == row["intended_label"]
    assert row["judge_model_id"] == SETTINGS.judge_model
    assert row["judge_request_id"] is not None
    assert json.loads(row["raw_sample"])["text"] == row["text"]
    answers = json.loads(row["judge_answers"])
    assert {"intent", "language_en"} <= set(answers)
    assert "development" not in answers
    assert min(a["noul"] for a in answers.values()) >= 0.5


def test_second_run_makes_no_calls_and_keeps_the_export(store, tmp_path):
    export = tmp_path / "export"
    run(store, FakeLlm(), export_dir=export)
    before = (export / "utterances.csv").read_bytes()
    client = FakeLlm()
    run(store, client, export_dir=export)
    assert client.calls["generate"] == 0
    assert (export / "utterances.csv").read_bytes() == before


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
    rows = rejected(store)
    assert len(rows) == 72 and rows[0]["reason"] == "schema: unknown_label"
    assert rows[0]["text"] and json.loads(rows[0]["raw_sample"])["label"] == "ambiguous"


def test_duplicates_of_accepted_text_are_rejected(store):
    report = run(store, FakeLlm(fixed_text="Mm-hmm"), languages=["en"], limit=1)
    assert report.accepted == 1
    assert report.reasons["dedupe:duplicate"] == 2


def test_received_response_is_resumed_before_new_requests(store):
    target = TARGETS[0]
    user = prompts.render_utterance_request(CAT, {f: target[f] for f in TRACK.factors}, count=3)
    request_id = store.insert_request(
        kind=prompts.UTTERANCE_REQUEST_KIND,
        combination_id=target["combination_id"],
        language=target["language"],
        requested_count=3,
        model_id=SETTINGS.generator_model,
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


def test_export_then_import_rebuilds_every_sample(store, tmp_path):
    run(store, FakeLlm(judge_no=("intent",)), languages=["en"], limit=2)
    store.export(tmp_path / "export")
    rebuilt = Store.open(tmp_path / "rebuilt.duckdb")
    try:
        rebuilt.import_exports(tmp_path / "export")
        assert rebuilt.utterances(status=None) == store.utterances(status=None)
        assert len(rebuilt.utterances(status="rejected")) == 6
    finally:
        rebuilt.close()


def test_code_switching_fits_inside_the_utterance():
    plan = build_utterance_plan(CAT, TRACK, replace(PLAN, criteria_per_language=200))
    assert {t.spec["code_switching"] for t in plan} == {"none", "mixed"}


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
    return run(store, client, replace(SETTINGS, concurrency=concurrency), **kwargs)


def test_parallel_run_overlaps_up_to_concurrency_and_fills_exactly(store):
    client = SlowLlm()
    report = run_parallel(store, client, concurrency=4, languages=["en"])
    assert client.max_active == 4
    assert client.calls["generate"] == 12
    assert report.requests == 12
    assert report.accepted == 36
    assert set(filled(store, EN_TARGETS)) == {3}


def test_parallel_run_dedupes_across_concurrent_responses(store, tmp_path):
    sequential = Store.open(tmp_path / "sequential.duckdb")
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
    unfilled = sum(1 for n in filled(store, EN_TARGETS) if n == 0)
    assert rejecting.calls["generate"] == 2 * unfilled  # two requests each, then the budget
    assert report.accepted == 0


# ---------------------------------------------------------------- judge


def test_a_criterion_the_judge_says_no_to_is_rejected_by_its_key(store):
    client = FakeLlm(judge_no=("language_",))
    report = run(store, client, languages=["en"])
    assert report.accepted == 0
    assert report.rejections["judge"] == 72
    assert report.reasons["judge:language_en"] == 72
    assert client.calls["generate"] == 24  # the attempt budget stops the retries
    row = rejected(store)[0]
    assert row["judge_request_id"] is not None
    assert row["reason"] == "judge: language_en=0.1"
    assert json.loads(row["judge_answers"])["language_en"]["noul"] == 0.1


def test_judge_api_error_keeps_the_response_for_the_next_run(store):
    failing = FakeLlm(judge_fails=True)
    report = run(store, failing, languages=["en"], limit=2)
    assert report.accepted == 0 and report.api_errors > 0
    received = store.requests(kind=prompts.UTTERANCE_REQUEST_KIND, status="received")
    assert len(received) == 2
    client = FakeLlm()
    run(store, client, languages=["en"], limit=0)
    assert client.calls["generate"] == 0
    assert client.calls["judge"] == 6
    assert sum(filled(store)) == 6


@pytest.mark.parametrize(
    ("template", "model"),
    [("generator_utterances_v1", None), (None, "an-earlier-model")],
)
def test_attempt_budget_counts_only_the_current_template_and_model(store, template, model):
    target = TARGETS[0]
    request_id = store.insert_request(
        kind=prompts.UTTERANCE_REQUEST_KIND,
        combination_id=target["combination_id"],
        language=target["language"],
        requested_count=6,
        model_id=model or SETTINGS.generator_model,
        prompt_template=template or prompts.UTTERANCE_TEMPLATE,
        prompt_hash="h",
    )
    store.fail_request(request_id, status="schema_invalid", error="earlier attempt")
    run(store, FakeLlm())
    assert set(filled(store)) == {3}


def test_all_combinations_plan_covers_every_valid_combination_equally():
    full = replace(PLAN, languages=["hi"], all_combinations=True)
    plan = build_utterance_plan(CAT, TRACK, full)
    ids_ = [t.combination_id for t in plan]
    assert len(ids_) == len(set(ids_))
    assert {t.required_count for t in plan} == {PLAN.required_count}
    valid = {
        ids.utterance_combination_id(spec)
        for option in CAT.language_options("hi")
        for intent in TRACK.intents
        for spec in _Builder(CAT, TRACK, full).valid_specs("hi", option[0], intent)
    }
    assert set(ids_) == valid
    sample = build_utterance_plan(CAT, TRACK, replace(PLAN, languages=["hi"]))
    assert {t.combination_id for t in sample} <= set(ids_)
    assert plan == build_utterance_plan(CAT, TRACK, full)
    for t in plan:
        assert not CAT.violations(t.spec, partial_spec=True)


def test_a_plan_can_limit_factor_values():
    english = replace(
        PLAN, languages=["en"], all_combinations=True, only={"code_switching": ["none"]}
    )
    plan = build_utterance_plan(CAT, TRACK, english)
    assert plan
    assert {t.spec["code_switching"] for t in plan} == {"none"}
    assert {t.spec["language"] for t in plan} == {"en"}
    full = build_utterance_plan(CAT, TRACK, replace(english, only={}))
    assert {t.combination_id for t in plan} < {t.combination_id for t in full}


def test_dry_run_plans_and_estimates_without_calls(store):
    client = FakeLlm()
    settings = replace(
        SETTINGS,
        prices_usd_per_million_tokens={
            SETTINGS.generator_model: {"input": 1.0, "cached_input": 0.1, "output": 2.0},
            SETTINGS.judge_model: {"input": 0.1, "cached_input": 0.0, "output": 0.0},
        },
    )
    report = run(store, client, settings, dry_run=True)
    assert report.planned_requests == 24 and report.planned_samples == 72
    assert report.estimated_cost_usd > 0
    assert client.calls == {"generate": 0, "judge": 0}
