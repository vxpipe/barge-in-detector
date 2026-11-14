from dataclasses import replace
from pathlib import Path

import pytest
from fake_llm import FakeLlm

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation.config import GenerationConfig
from barge_in_detector.generation.plan import PlanConfig, build_plan
from barge_in_detector.generation.runner import Runner
from barge_in_detector.storage.store import Store

CAT = load_catalogue()
PLAN = PlanConfig(
    name="runner-test",
    seed=11,
    languages=["en"],
    required_count=3,
    label_shares={"backchannel": 0.5, "barge_in": 0.5},
    criteria_per_language=4,
    contrastive_instances_per_group=0,
    candidates_per_pick=2,
)
SETTINGS = GenerationConfig(batch_size=10, max_attempts_per_sample=2, concurrency=2)


@pytest.fixture
def store(tmp_path: Path):
    s = Store.open(tmp_path / "work.duckdb")
    s.apply_plan(CAT, build_plan(CAT, PLAN))
    yield s
    s.close()


def run(store, client, **kwargs):
    return Runner(store, CAT, client, SETTINGS).run(**kwargs)


def filled(store):
    counts = store.accepted_counts()
    return {t["combination_id"]: counts.get(t["combination_id"], 0) for t in store.targets()}


def files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_run_fills_every_criterion_to_its_count(store):
    client = FakeLlm()
    report = run(store, client)
    assert set(filled(store).values()) == {3}
    assert client.calls["generate"] == 4  # one request of 3 per criterion
    assert report.accepted == 12
    assert len(client.system_prompts) == 2  # one generator and one judge system prompt


def test_second_run_makes_no_calls_and_keeps_the_export(store, tmp_path):
    run(store, FakeLlm(), export_dir=tmp_path / "export")
    before = files(tmp_path / "export")
    texts = {c["conversation_id"]: c["candidate_text"] for c in store.conversations()}
    client = FakeLlm()
    run(store, client, export_dir=tmp_path / "export")
    assert client.calls == {"generate": 0, "judge": 0}
    assert files(tmp_path / "export") == before
    assert store.count("runs") == 1  # a run with nothing to do is not recorded
    assert {c["conversation_id"]: c["candidate_text"] for c in store.conversations()} == texts


def test_limited_run_then_another_fills_exactly_the_rest(store):
    run(store, FakeLlm(), limit=2)
    assert sorted(filled(store).values()) == [0, 0, 3, 3]
    client = FakeLlm()
    run(store, client)
    assert client.calls["generate"] == 2
    assert set(filled(store).values()) == {3}


def test_raising_a_count_generates_only_the_difference(store):
    run(store, FakeLlm())
    store.apply_plan(CAT, build_plan(CAT, replace(PLAN, required_count=5)))
    client = FakeLlm()
    run(store, client)
    assert client.calls["generate"] == 4
    assert set(filled(store).values()) == {5}
    requested = store.con.execute(
        "SELECT requested_count FROM requests WHERE kind = 'generate' ORDER BY request_id DESC LIMIT 4"
    ).fetchall()
    assert {r[0] for r in requested} == {2}


def test_interrupted_run_resumes_without_new_generation(store):
    run(store, FakeLlm(judge_fails=True))
    assert set(filled(store).values()) == {0}
    assert len(store.requests(kind="generate", status="received")) == 4
    client = FakeLlm()
    run(store, client)
    assert client.calls["generate"] == 0
    assert set(filled(store).values()) == {3}
    assert store.requests(kind="generate", status="received") == []


def test_retired_conversation_is_replaced_and_kept(store):
    run(store, FakeLlm())
    victim = store.conversations()[0]
    store.retire_conversation(victim["conversation_id"], reason="review")
    client = FakeLlm()
    run(store, client)
    assert client.calls["generate"] == 1
    assert set(filled(store).values()) == {3}
    kept = store.conversation(victim["conversation_id"])
    assert kept["status"] == "retired" and kept["candidate_text"] == victim["candidate_text"]


def test_label_disagreement_is_rejected_until_the_budget_runs_out(store):
    client = FakeLlm(judge_disagrees=True)
    report = run(store, client)
    assert set(filled(store).values()) == {0}
    assert report.rejections["label_mismatch"] == 4 * 3 * 2  # budget: 2 attempts per sample
    assert len(report.exhausted) == 4
    rows = store.con.execute("SELECT DISTINCT stage FROM rejections").fetchall()
    assert rows == [("label_mismatch",)]


def test_api_errors_are_recorded_and_retried_next_run(store):
    run(store, FakeLlm(generator_fails=True), limit=1)
    assert store.requests(kind="generate", status="api_error")
    run(store, FakeLlm())
    assert set(filled(store).values()) == {3}


def test_dry_run_writes_nothing(store):
    client = FakeLlm()
    report = run(store, client, dry_run=True)
    assert client.calls == {"generate": 0, "judge": 0}
    assert store.count("requests") == 0 and store.count("runs") == 0
    assert report.planned_requests == 4
    assert report.planned_samples == 12


def test_contrastive_groups_fill_in_lockstep(tmp_path):
    s = Store.open(tmp_path / "c.duckdb")
    s.apply_plan(
        CAT,
        build_plan(CAT, replace(PLAN, criteria_per_language=0, contrastive_instances_per_group=1)),
    )
    client = FakeLlm()
    Runner(s, CAT, client, SETTINGS).run()
    counts = filled(s)
    assert set(counts.values()) == {3}
    assert client.calls["generate"] == len(CAT.contrastive)
    s.close()
