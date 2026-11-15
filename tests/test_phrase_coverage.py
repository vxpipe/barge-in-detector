import json

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation.llm import LlmError, LlmResult
from barge_in_detector.generation.phrase_coverage import (
    PLAN_PATH,
    build_targets,
    load_plan,
    sample_problem,
)
from barge_in_detector.generation.phrase_runner import PhraseRunner, request_reserve
from barge_in_detector.storage.store import Store


def sample(text="Okay, I follow you.", anchor="Okay"):
    return {"text": text, "anchor": anchor, "label": "backchannel", "rationale": "Acknowledgment"}


class FakePhrases:
    def __init__(self, samples=None, fail=False):
        self.samples = samples or [sample()]
        self.calls = 0
        self.fail = fail

    def generate(self, **kwargs):
        self.calls += 1
        assert kwargs["max_output_tokens"] > 0
        if self.fail:
            raise LlmError("Lost response after submission")
        text = json.dumps({"samples": self.samples})
        usage = {"input_tokens": 100, "output_tokens": 100, "cached_input_tokens": 0}
        return LlmResult(text, {"text": text, "usage": usage}, usage)

    def decide(self, **kwargs):
        raise AssertionError("Phrase generation must not call a judge")


def setup(tmp_path, *, budget=8):
    cat, plan = load_catalogue(), load_plan(PLAN_PATH)
    plan["new_budget_usd"] = budget
    plan["required_count"] = 1
    targets = build_targets(cat, plan)
    target = next(
        t
        for t in targets
        if t["language"] == "en"
        and t["regional_variant"] == "en-US"
        and t["coverage_form"] == "expanded"
        and t["caller_intent"] == "acknowledgment"
        and t["code_switching"] == "none"
    )
    return cat, plan, target, Store.open(tmp_path / "phrases.duckdb")


def test_plan_covers_manifest_native_scripts_and_keeps_ids_when_counts_grow():
    cat, plan = load_catalogue(), load_plan(PLAN_PATH)
    targets = build_targets(cat, plan)
    assert {t["language"] for t in targets} == set(cat.languages)
    assert {t["regional_variant"] for t in targets if t["language"] == "en"} == {
        "en-US",
        "en-IN",
        "en-SG",
    }
    assert all(t["script_form"] == "native" for t in targets)
    assert {t["label"] for t in targets} == {"backchannel", "barge_in"}
    assert all(t["code_switching"] == "none" for t in targets if t["coverage_form"] == "core")
    larger = build_targets(cat, {**plan, "required_count": plan["required_count"] + 2})
    assert [t["combination_id"] for t in targets] == [t["combination_id"] for t in larger]
    assert len({t["combination_id"] for t in targets}) == len(targets)


def test_fill_resume_and_noop_preserve_export_without_a_judge(tmp_path):
    cat, plan, target, store = setup(tmp_path)
    client = FakePhrases()
    try:
        runner = PhraseRunner(store, cat, plan, [target], client, excluded={})
        runner.run(tmp_path / "export")
        rows = store.utterances()
        assert len(rows) == 1 and rows[0]["judge_request_id"] is None
        assert rows[0]["judge_model_id"] is None
        raw = json.loads(rows[0]["raw_sample"])
        assert raw["coverage"]["coverage_form"] == "expanded"
        export = tmp_path / "export" / "utterances.csv"
        before = export.read_bytes(), export.stat().st_mtime_ns
        assert not runner.needs_generation()
        runner.run(tmp_path / "export")
        assert client.calls == 1
        assert before == (export.read_bytes(), export.stat().st_mtime_ns)
    finally:
        store.close()


def test_budget_is_checked_before_submission_and_unknown_cost_survives_restart(tmp_path):
    cat, plan, target, store = setup(tmp_path, budget=0)
    client = FakePhrases(fail=True)
    try:
        runner = PhraseRunner(store, cat, plan, [target], client, excluded={})
        runner.run(tmp_path / "export")
        assert client.calls == store.count("requests") == 0
        plan["new_budget_usd"] = request_reserve(plan) * 1.5
        runner.run(tmp_path / "export")
        assert client.calls == 1
        assert runner.committed_cost() == pytest.approx(request_reserve(plan))
    finally:
        store.close()
    store = Store.open(tmp_path / "phrases.duckdb")
    try:
        PhraseRunner(store, cat, plan, [target], client, excluded={}).run(tmp_path / "export")
        assert client.calls == 1
    finally:
        store.close()


def test_received_request_is_processed_without_repeating_the_paid_call(tmp_path):
    cat, plan, target, store = setup(tmp_path)
    client = FakePhrases()
    try:
        runner = PhraseRunner(store, cat, plan, [target], client, excluded={})
        rid = runner.submit_record(target, 1)
        result = client.generate(max_output_tokens=100)
        store.record_response(rid, response=result.response, usage=result.usage, cost_usd=0.001)
        runner.run(tmp_path / "export")
        assert client.calls == 1 and len(store.utterances()) == 1
    finally:
        store.close()


def test_duplicate_and_reserved_evaluation_texts_never_enter_supplement(tmp_path):
    cat, plan, target, store = setup(tmp_path)
    client = FakePhrases()
    try:
        PhraseRunner(
            store,
            cat,
            plan,
            [target],
            client,
            excluded={"okay i follow you": "reserved_evaluation"},
        ).run(tmp_path / "export")
        assert not store.utterances()
        assert all(r["reason"] == "reserved_evaluation" for r in store.utterances(status=None))
        assert client.calls == plan["max_attempts"]
    finally:
        store.close()


def test_deterministic_checks_reject_wrong_script_and_fake_repetitions(tmp_path):
    cat, plan, target, store = setup(tmp_path)
    store.close()
    repetition = {**target, "coverage_form": "repetition"}
    assert sample_problem(cat, repetition, sample("Okay okay", "Okay")) is None
    assert sample_problem(cat, repetition, sample()) == "not_a_repetition"
    hindi = next(
        t
        for t in build_targets(cat, plan)
        if t["language"] == "hi" and t["code_switching"] == "none" and t["coverage_form"] == "core"
    )
    assert sample_problem(cat, hindi, sample("Haan", "Haan")) == "wrong_script"
    assert sample_problem(cat, hindi, sample("हाँ", "हाँ")) is None


def test_parallel_requests_cannot_overbook_the_budget(tmp_path):
    cat, plan, _, store = setup(tmp_path)
    plan["new_budget_usd"] = request_reserve(plan) * 1.5
    client = FakePhrases(fail=True)
    try:
        PhraseRunner(store, cat, plan, build_targets(cat, plan)[:10], client, excluded={}).run(
            tmp_path / "export"
        )
        assert client.calls == store.count("requests") == 1
    finally:
        store.close()


def test_authentication_failure_stops_before_sending_the_rest_of_the_plan(tmp_path):
    cat, plan, _, store = setup(tmp_path)
    plan["concurrency"] = 1

    class BadKey(FakePhrases):
        def generate(self, **kwargs):
            self.calls += 1
            raise LlmError("401: Invalid credential")

    client = BadKey()
    try:
        result = PhraseRunner(
            store, cat, plan, build_targets(cat, plan)[:10], client, excluded={}
        ).run(tmp_path / "export")
        assert client.calls == 1
        assert result["abort_reason"].startswith("401:")
    finally:
        store.close()


def test_raising_required_count_can_add_data_without_resetting_attempts(tmp_path):
    cat, plan, target, store = setup(tmp_path)

    class Distinct(FakePhrases):
        def generate(self, **kwargs):
            self.samples = [sample(f"Okay, I follow point {self.calls}.")]
            return super().generate(**kwargs)

    client = Distinct()
    try:
        for required in (1, 2, 3):
            target["required_count"] = required
            PhraseRunner(store, cat, plan, [target], client, excluded={}).run(tmp_path / "export")
        assert len(store.utterances()) == 3
        assert client.calls == 3
    finally:
        store.close()


def test_budget_refuses_a_model_without_matching_prices(tmp_path):
    cat, plan, target, store = setup(tmp_path)
    try:
        with pytest.raises(ValueError, match="prices"):
            PhraseRunner(
                store, cat, plan, [target], FakePhrases(), excluded={}, model="unpriced-model"
            )
    finally:
        store.close()
