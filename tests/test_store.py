import uuid
from pathlib import Path

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation.plan import PlanConfig, build_plan
from barge_in_detector.storage.store import RefusedUpdate, Store

CAT = load_catalogue()
CONFIG = PlanConfig(
    name="store-test",
    seed=3,
    languages=["hi"],
    required_count=2,
    label_shares={"backchannel": 0.5, "barge_in": 0.5},
    criteria_per_language=6,
    contrastive_instances_per_group=0,
    candidates_per_pick=2,
)


@pytest.fixture
def store(tmp_path: Path):
    s = Store.open(tmp_path / "work" / "test.duckdb")
    s.apply_plan(CAT, build_plan(CAT, CONFIG))
    yield s
    s.close()


def files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_apply_plan_inserts_targets_once(store):
    assert len(store.targets()) == 6
    store.apply_plan(CAT, build_plan(CAT, CONFIG))
    assert len(store.targets()) == 6
    assert store.count("plan_revisions") == 1  # an unchanged plan records nothing


def test_raising_required_count_updates_existing_targets(store):
    store.apply_plan(
        CAT, build_plan(CAT, CONFIG.__class__(**{**CONFIG.__dict__, "required_count": 5}))
    )
    assert {t["required_count"] for t in store.targets()} == {5}


def test_response_is_recorded_once(store):
    run_id = store.start_run(
        name="r", revision_id=None, settings={}, generator_model_id="g", judge_model_id="j"
    )
    target = store.targets()[0]
    request_id = store.insert_request(
        run_id=run_id,
        kind="generate",
        combination_id=target["combination_id"],
        contrastive_group_id=None,
        language="hi",
        requested_count=2,
        model_id="g",
        prompt_template="t",
        prompt_hash="h",
    )
    store.record_response(request_id, response={"text": "{}"}, usage={}, cost_usd=None)
    with pytest.raises(RefusedUpdate):
        store.record_response(request_id, response={"text": "changed"}, usage={}, cost_usd=None)
    assert store.request(request_id)["status"] == "received"


def test_retiring_a_conversation_keeps_its_text(store):
    run_id = store.start_run(
        name="r", revision_id=None, settings={}, generator_model_id="g", judge_model_id="j"
    )
    target = store.targets()[0]
    conversation_id = store.insert_conversation(fake_conversation(target, run_id))
    store.retire_conversation(conversation_id, reason="human review: wrong label")
    row = store.conversation(conversation_id)
    assert row["status"] == "retired"
    assert row["candidate_text"] == "haan haan"
    with pytest.raises(RefusedUpdate):
        store.retire_conversation(conversation_id, reason="again")


def fake_conversation(target, run_id):
    return {
        "target": target,
        "request_id": uuid.uuid7(),
        "judge_request_id": uuid.uuid7(),
        "run_id": run_id,
        "sample_index": 0,
        "turns": [{"speaker": "Riya", "role": "caller", "text": "Mera card kho gaya."}],
        "agent_speaker": "Agent",
        "agent_full_text": "Main aapka card block kar rahi hoon. Naya card aayega.",
        "agent_spoken_text": "Main aapka card block",
        "candidate_speaker": "Riya",
        "candidate_role": "caller",
        "candidate_text": "haan haan",
        "gemini_label": "backchannel",
        "gemini_rationale": "agreement",
        "barge_in_from_word": None,
        "judge_label": "backchannel",
        "judge_barge_in_from_word": None,
        "judge_notes": "",
        "model_id": "g",
        "prompt_template": "t",
        "judge_model_id": "j",
        "text_hash": "abc",
    }


def test_export_is_deterministic_and_import_round_trips(store, tmp_path):
    run_id = store.start_run(
        name="r", revision_id=None, settings={"x": 1}, generator_model_id="g", judge_model_id="j"
    )
    store.insert_conversation(fake_conversation(store.targets()[0], run_id))
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    store.export(out_a)
    store.export(out_b)
    assert files(out_a) and files(out_a) == files(out_b)
    assert any(name.startswith("targets/language=hi/") for name in files(out_a))

    rebuilt = Store.open(tmp_path / "rebuilt.duckdb")
    rebuilt.import_exports(out_a)
    out_c = tmp_path / "c"
    rebuilt.export(out_c)
    rebuilt.close()
    assert files(out_c) == files(out_a)


def test_export_leaves_unchanged_files_alone(store, tmp_path):
    out = tmp_path / "export"
    store.export(out)
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*.parquet")}
    store.export(out)
    after = {p: p.stat().st_mtime_ns for p in out.rglob("*.parquet")}
    assert before == after
