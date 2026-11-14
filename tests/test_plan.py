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


def test_contrastive_groups_are_spread_through_the_generation_order(cat, plan):
    order = sorted(plan.targets, key=lambda t: (t.priority, t.quota_group, str(t.combination_id)))
    first = order[: len(order) // 4]
    assert {t.quota_group for t in first} == {"stratified", "contrastive"}
    contrastive = [t.priority for t in plan.targets if t.quota_group == "contrastive"]
    assert max(contrastive) >= CONFIG.criteria_per_language // 2


def test_variant_shares_are_followed(cat):
    config = replace(
        CONFIG,
        languages=["en"],
        criteria_per_language=80,
        contrastive_instances_per_group=0,
        variant_shares={"en": {"en-US": 0.5, "en-IN": 0.5}},
    )
    plan = build_plan(cat, config)
    variants = Counter(t.spec["regional_variant"] for t in plan.targets)
    assert variants == {"en-US": 40, "en-IN": 40}
    # en-US has no code-switching partner: always plain English.
    assert {
        t.spec["code_switching"] for t in plan.targets if t.spec["regional_variant"] == "en-US"
    } == {"none"}


def test_code_switching_is_chosen_before_the_set(cat):
    config = replace(
        CONFIG,
        languages=["en"],
        criteria_per_language=90,
        contrastive_instances_per_group=0,
        variant_shares={"en": {"en-IN": 1.0}},
    )
    switching = Counter(t.spec["code_switching"] for t in build_plan(cat, config).targets)
    # 1 of 55 en-IN options has no code-switching; choosing it first gives about a third.
    assert switching["none"] >= 15


def test_variants_default_to_equal_shares(cat):
    config = replace(
        CONFIG, languages=["en"], criteria_per_language=100, contrastive_instances_per_group=0
    )
    variants = Counter(t.spec["regional_variant"] for t in build_plan(cat, config).targets)
    assert set(variants.values()) == {10}


def test_pilot_keeps_plain_indian_english(cat):
    pilot = PlanConfig.from_file(
        Path(__file__).resolve().parents[1] / "generation/plans/en-hi-pilot.yaml"
    )
    en_in = [t for t in build_plan(cat, pilot).targets if t.spec["regional_variant"] == "en-IN"]
    plain = sum(t.spec["code_switching"] == "none" for t in en_in)
    assert plain / len(en_in) >= 0.20  # drawn at 1/3; random spread on ~50 criteria


def test_contrastive_group_ids_depend_on_their_members(cat, plan):
    other = build_plan(cat, replace(CONFIG, seed=CONFIG.seed + 1))
    ids_a = {t.contrastive_group_id for t in plan.targets if t.contrastive_group_id}
    ids_b = {t.contrastive_group_id for t in other.targets if t.contrastive_group_id}
    assert ids_a and not ids_a & ids_b  # a rebuilt plan never reuses a group id for other members
    again = build_plan(cat, CONFIG)
    assert ids_a == {t.contrastive_group_id for t in again.targets if t.contrastive_group_id}


def test_mixed_in_chinese_is_drawn_both_ways(cat):
    config = replace(
        CONFIG,
        languages=["en"],
        criteria_per_language=60,
        contrastive_instances_per_group=0,
        variant_shares={"en": {"en-MY": 1.0}},
    )
    scripts = Counter(
        t.spec["mixed_in_script"]
        for t in build_plan(cat, config).targets
        if t.spec["code_switching"] != "none"
    )
    assert scripts["native"] >= 10 and scripts["romanized"] >= 10


def test_split_audit_tolerance_is_inclusive(cat, plan):
    # A share exactly at the tolerance edge (e.g. 0.08 vs 0.10) is within it.
    audit = split_audit(cat, plan.targets, CONFIG.split_ratios, tolerance=0.02, min_support=1)
    assert not any(f.endswith(": 0.080 vs 0.100") for f in audit.failures)
