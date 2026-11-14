from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation.plan import (
    ExistingTarget,
    PlanConfig,
    build_plan,
    split_audit,
)


@pytest.fixture(scope="module")
def cat():
    return load_catalogue()


CONFIG = PlanConfig(
    name="test",
    seed=7,
    languages=["en", "hi"],
    required_count=10,
    label_shares={"backchannel": 0.45, "barge_in": 0.5, "ambiguous": 0.05},
    criteria_per_language=60,
    contrastive_instances_per_group=1,
    candidates_per_pick=6,
    split_ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    heldout={"domain": ["agri_rural"]},
)


@pytest.fixture(scope="module")
def plan(cat):
    return build_plan(cat, CONFIG)


def test_same_inputs_give_the_same_plan(cat, plan):
    again = build_plan(cat, CONFIG)
    assert [t.combination_id for t in again.targets] == [t.combination_id for t in plan.targets]
    assert [t.split for t in again.targets] == [t.split for t in plan.targets]


def test_every_criterion_is_valid_and_unique(cat, plan):
    assert all(cat.is_valid(t.spec) for t in plan.targets)
    ids = [t.combination_id for t in plan.targets]
    assert len(ids) == len(set(ids))


def test_sizes_follow_the_config(cat, plan):
    counts = Counter(t.quota_group for t in plan.targets)
    assert counts["stratified"] == 2 * 60
    members = sum(len(g["members"]) for g in cat.contrastive)
    assert counts["contrastive"] == 2 * members
    assert {t.required_count for t in plan.targets} == {10}


def test_label_shares_are_followed(plan):
    labels = Counter(t.intended_label for t in plan.targets if t.quota_group == "stratified")
    assert labels["backchannel"] == pytest.approx(0.45 * 120, abs=2)
    assert labels["barge_in"] == pytest.approx(0.5 * 120, abs=2)


def test_plan_is_prefix_stable(cat, plan):
    bigger = build_plan(
        cat, replace(CONFIG, criteria_per_language=90, contrastive_instances_per_group=2)
    )
    small_ids = {t.combination_id for t in plan.targets}
    big_ids = {t.combination_id for t in bigger.targets}
    assert small_ids <= big_ids


def test_contrastive_members_share_everything_but_their_overrides(cat, plan):
    by_group: dict[str, list] = {}
    for t in plan.targets:
        if t.contrastive_group_id:
            by_group.setdefault(t.contrastive_group_id, []).append(t)
    assert by_group
    for group_id, members in by_group.items():
        group = next(g for g in cat.contrastive if group_id.startswith(g["id"] + ":"))
        varied = {k for m in group["members"] for k in m}
        shared = [f for f in cat.spec_factors if f not in varied]
        assert len({tuple(m.spec[f] for f in shared) for m in members}) == 1
        assert len({m.split for m in members}) == 1
        assert len({m.split_group_id for m in members}) == 1


def test_heldout_values_never_reach_other_splits(plan):
    for t in plan.targets:
        if t.spec["domain"] == "agri_rural":
            assert t.split == "heldout"
        else:
            assert t.split != "heldout"


def test_validation_and_test_values_appear_in_train(cat, plan):
    train = {(f, t.spec[f]) for t in plan.targets if t.split == "train" for f in cat.spec_factors}
    for t in plan.targets:
        if t.split in ("validation", "test"):
            assert {(f, t.spec[f]) for f in cat.spec_factors} <= train


def test_existing_targets_keep_their_split(cat, plan):
    first = plan.targets[0]
    flipped = "test" if first.split != "test" else "validation"
    existing = [
        ExistingTarget(
            combination_id=first.combination_id,
            split=flipped,
            split_group_id=first.split_group_id,
            quota_group=first.quota_group,
            required_count=first.required_count,
        )
    ]
    again = build_plan(cat, CONFIG, existing=existing)
    assert again.targets[0].split == flipped


def test_split_audit_reports_label_shares(cat, plan):
    audit = split_audit(cat, plan.targets, CONFIG.split_ratios, min_support=10_000)
    assert audit.failures == []
    assert any(row.feature == "label=backchannel" for row in audit.rows)


def test_pilot_plan_passes_the_split_audit(cat):
    pilot = PlanConfig.from_file(
        Path(__file__).resolve().parents[1] / "generation/plans/en-hi-pilot.yaml"
    )
    plan = build_plan(cat, pilot)
    audit = split_audit(
        cat,
        plan.targets,
        pilot.split_ratios,
        tolerance=pilot.split_tolerance,
        min_support=pilot.split_min_support,
    )
    assert audit.failures == []
