"""Deterministic plan builder: from (catalogue, plan config, seed) to criteria.

Two streams per language, both prefix-stable (the criteria chosen for a size N
are always the first N chosen for any larger size):

- stratified: cells of (label, intent) interleaved by label share; within a cell
  each pick is the candidate spec that adds the most uncovered factor-value
  pairs (greedy all-pairs coverage).
- contrastive: instances of each contrastive group; members share every factor
  their overrides do not set.

Splits are assigned per split group, in seeded hash order, by iterative
stratification; see docs/data-format.md#splits.
"""

import hashlib
import itertools
import random
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from barge_in_detector import ids
from barge_in_detector.catalogue import LANGUAGE_FACTORS, Catalogue

SPLITS = ("train", "validation", "test")
_MAX_SAMPLE_TRIES = 400


@dataclass(frozen=True)
class PlanConfig:
    name: str
    seed: int
    languages: list[str]
    required_count: int
    label_shares: dict[str, float]
    criteria_per_language: int
    contrastive_instances_per_group: int
    candidates_per_pick: int = 12
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    heldout: dict[str, list[str]] = field(default_factory=dict)
    # language -> {regional_variant: share}; variants left out get none.
    # Languages not listed spread their criteria equally over their variants.
    variant_shares: dict[str, dict[str, float]] = field(default_factory=dict)
    split_tolerance: float = 0.02
    split_min_support: int = 50

    @classmethod
    def from_file(cls, path: Path) -> PlanConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def as_record(self) -> dict[str, Any]:
        return {
            "quotas": {
                "languages": self.languages,
                "required_count": self.required_count,
                "label_shares": self.label_shares,
                "criteria_per_language": self.criteria_per_language,
                "contrastive_instances_per_group": self.contrastive_instances_per_group,
                "candidates_per_pick": self.candidates_per_pick,
                "variant_shares": self.variant_shares,
            },
            "split_ratios": self.split_ratios,
            "heldout": self.heldout,
        }


@dataclass(frozen=True)
class ExistingTarget:
    combination_id: uuid.UUID
    split: str
    split_group_id: uuid.UUID
    quota_group: str
    required_count: int


@dataclass(frozen=True)
class PlannedTarget:
    spec: dict[str, str]
    combination_id: uuid.UUID
    agent_state: str
    intended_label: str
    required_count: int
    quota_group: str
    priority: int
    contrastive_group_id: str | None
    contrastive_role: int | None
    split_group_id: uuid.UUID
    split: str


@dataclass(frozen=True)
class Plan:
    config: PlanConfig
    targets: list[PlannedTarget]


# ---------------------------------------------------------------- sampling


class _Sampler:
    def __init__(self, cat: Catalogue):
        self.cat = cat
        self._language_options: dict[str, list[tuple[str, str, str, str]]] = {}
        self._order = [
            f for f in cat.spec_factors if f not in (*LANGUAGE_FACTORS, "domain", "scenario")
        ]

    def language_options(self, language: str) -> list[tuple[str, str, str, str]]:
        if language not in self._language_options:
            self._language_options[language] = self.cat.language_options(language)
        return self._language_options[language]

    def sample(
        self,
        rng: random.Random,
        fixed: Mapping[str, str],
        variant_weights: Mapping[str, float] | None = None,
    ) -> dict[str, str] | None:
        """One random valid spec with the `fixed` values, or None."""
        cat = self.cat
        for _ in range(_MAX_SAMPLE_TRIES):
            spec = dict(fixed)
            options = [
                o
                for o in self.language_options(spec["language"])
                if all(spec.get(f, v) == v for f, v in zip(LANGUAGE_FACTORS[1:], o, strict=True))
            ]
            if not options:
                return None
            # Choose the language side one factor at a time (variant, script,
            # code-switching, then the set), so a value with many options under
            # it (en-IN's code-switch sets) does not crowd out the rest.
            for position in range(len(LANGUAGE_FACTORS) - 1):
                values = sorted({o[position] for o in options})
                weights = None
                if position == 0 and variant_weights:
                    weights = [variant_weights.get(v, 0.0) for v in values]
                    if not any(weights):
                        weights = None
                chosen = rng.choices(values, weights=weights)[0] if weights else rng.choice(values)
                options = [o for o in options if o[position] == chosen]
            spec.update(zip(LANGUAGE_FACTORS[1:], options[0], strict=True))
            if "scenario" not in spec:
                scenarios = [
                    s
                    for s, info in cat.scenarios.items()
                    if spec.get("domain", info["domain"]) == info["domain"]
                ]
                spec["scenario"] = rng.choice(scenarios)
            spec["domain"] = cat.scenarios[spec["scenario"]]["domain"]
            for factor in self._order:
                if factor in spec:
                    continue
                allowed = cat.allowed_values(factor, spec)
                if not allowed:
                    break
                spec[factor] = rng.choice(allowed)
            else:
                if cat.is_valid(spec):
                    return {f: spec[f] for f in cat.spec_factors}
        return None


def _features(spec: Mapping[str, str], factors: Sequence[str]) -> list[str]:
    return [f"{f}={spec[f]}" for f in factors]


class _Coverage:
    def __init__(self, factors: Sequence[str]):
        self.factors = list(factors)
        self.singles: Counter[str] = Counter()
        self.pairs: Counter[tuple[str, str]] = Counter()

    def score(self, spec: Mapping[str, str]) -> float:
        feats = _features(spec, self.factors)
        return sum(1 / (1 + self.singles[f]) for f in feats) + sum(
            1 / (1 + self.pairs[p]) for p in itertools.combinations(feats, 2)
        )

    def add(self, spec: Mapping[str, str]) -> None:
        feats = _features(spec, self.factors)
        self.singles.update(feats)
        self.pairs.update(itertools.combinations(feats, 2))


def _cells(cat: Catalogue, label_shares: Mapping[str, float]) -> Iterable[str]:
    """Infinite, deterministic sequence of intents following the label shares."""
    by_label = {
        label: [i for i, data in cat.intents.items() if data["label"] == label]
        for label in label_shares
        if label_shares[label] > 0
    }
    label_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    labels = list(by_label)
    while True:
        label = min(labels, key=lambda lb: (label_counts[lb] / label_shares[lb], labels.index(lb)))
        intents = by_label[label]
        intent = min(intents, key=lambda i: (intent_counts[i], intents.index(i)))
        label_counts[label] += 1
        intent_counts[intent] += 1
        yield intent


def _variant_shares(cat: Catalogue, config: PlanConfig, language: str) -> dict[str, float]:
    """The plan's shares for the language's regional variants; equal by default."""
    variants = cat.variants(language)
    shares = config.variant_shares.get(language)
    if not shares:
        return {v: 1.0 for v in variants}
    unknown = set(shares) - set(variants)
    if unknown:
        raise ValueError(f"unknown variants for {language}: {sorted(unknown)}")
    return {v: float(shares.get(v, 0.0)) for v in variants if shares.get(v, 0.0) > 0}


def _variant_cells(shares: Mapping[str, float]) -> Iterable[str]:
    """Infinite, deterministic sequence of variants following the shares."""
    order = list(shares)
    counts: Counter[str] = Counter()
    while True:
        variant = min(order, key=lambda v: (counts[v] / shares[v], order.index(v)))
        counts[variant] += 1
        yield variant


def _stratified(
    cat: Catalogue,
    sampler: _Sampler,
    config: PlanConfig,
    language: str,
    skip: set[uuid.UUID],
) -> list[tuple[int, dict[str, str]]]:
    rng = random.Random(f"{config.seed}:stratified:{language}")
    # The language side (variant by share, then script, code-switching and set,
    # each drawn evenly) is not scored for coverage: rare code-switch sets, and
    # "none" being common across other variants, would otherwise make
    # code-switched candidates win every pick.
    coverage = _Coverage([f for f in cat.spec_factors if f not in LANGUAGE_FACTORS])
    picked: list[tuple[int, dict[str, str]]] = []
    taken: set[uuid.UUID] = set()
    variants = _variant_cells(_variant_shares(cat, config, language))
    # Both sequences are infinite.
    cells = zip(_cells(cat, config.label_shares), variants, strict=True)
    for step, (intent, variant) in enumerate(cells):
        if len(picked) >= config.criteria_per_language:
            break
        best: dict[str, str] | None = None
        best_score = -1.0
        for _ in range(config.candidates_per_pick):
            spec = sampler.sample(
                rng, {"language": language, "regional_variant": variant, "caller_intent": intent}
            )
            if spec is None:
                continue
            cid = ids.combination_id(spec)
            if cid in taken or cid in skip:
                continue
            score = coverage.score(spec)
            if score > best_score:
                best, best_score = spec, score
        if best is None:
            continue
        coverage.add(best)
        taken.add(ids.combination_id(best))
        picked.append((step, best))
    return picked


def _contrastive(
    cat: Catalogue,
    sampler: _Sampler,
    config: PlanConfig,
    language: str,
    skip: set[uuid.UUID],
) -> list[tuple[int, str, list[dict[str, str]]]]:
    instances = []
    taken: set[uuid.UUID] = set()
    for position, group in enumerate(cat.contrastive):
        rng = random.Random(f"{config.seed}:contrastive:{language}:{group['id']}")
        members = group["members"]
        for index in range(config.contrastive_instances_per_group):
            for _ in range(_MAX_SAMPLE_TRIES):
                base = sampler.sample(
                    rng,
                    {"language": language, **members[0]},
                    variant_weights=_variant_shares(cat, config, language),
                )
                if base is None:
                    continue
                specs = [{**base, **{k: str(v) for k, v in m.items()}} for m in members]
                member_ids = [ids.combination_id(s) for s in specs]
                if all(cat.is_valid(s) for s in specs) and not (set(member_ids) & (taken | skip)):
                    taken.update(member_ids)
                    instances.append(
                        (
                            _spread_priority(config, len(cat.contrastive), position, index),
                            # Unique to these members, so a rebuilt plan never
                            # reuses an id that earlier requests point at.
                            f"{group['id']}:{language}:{ids.split_group_id(member_ids).hex[:12]}",
                            specs,
                        )
                    )
                    break
            else:
                raise ValueError(f"cannot sample contrastive group {group['id']} for {language}")
    return instances


def _spread_priority(config: PlanConfig, groups: int, group_position: int, index: int) -> int:
    """Spread contrastive instances evenly through the stratified generation order.

    Instance k (index-major, then group order) of K gets priority
    k * criteria_per_language // K, so a limited run sees both kinds. It
    depends only on the config, so it is deterministic and stable as plans grow
    with the same sizes.
    """
    total = max(1, groups * config.contrastive_instances_per_group)
    k = index * groups + group_position
    return k * max(1, config.criteria_per_language) // total


# ---------------------------------------------------------------- splits


def _is_heldout(spec: Mapping[str, str], heldout: Mapping[str, Sequence[str]]) -> bool:
    return any(spec.get(f) in values for f, values in heldout.items())


def _group_features(cat: Catalogue, members: Sequence[tuple[dict[str, str], str, int]]) -> Counter:
    weights: Counter[str] = Counter()
    for spec, label, count in members:
        for feat in [*_features(spec, cat.spec_factors), f"label={label}"]:
            weights[feat] += count
    return weights


def _assign_splits(
    cat: Catalogue,
    config: PlanConfig,
    groups: dict[uuid.UUID, list[tuple[dict[str, str], str, int]]],
    known: Mapping[uuid.UUID, str],
) -> dict[uuid.UUID, str]:
    """Splits for every group; groups in `known` keep theirs and count as assigned."""
    ratios = config.split_ratios
    assigned: dict[str, Counter] = {s: Counter() for s in SPLITS}
    total: Counter[str] = Counter()

    result: dict[uuid.UUID, str] = {}
    pending = []
    for gid, members in groups.items():
        if gid in known:
            result[gid] = known[gid]
            if known[gid] in assigned:
                weights = _group_features(cat, members)
                assigned[known[gid]].update(weights)
                total.update(weights)
        elif any(_is_heldout(spec, config.heldout) for spec, _, _ in members):
            result[gid] = "heldout"
        else:
            pending.append(gid)
    # Iterative stratification: rarest groups first (by the rarest feature they
    # carry among the pending groups), so groups forced into train by an unseen
    # value come early and the common values, placed last, balance the splits.
    # Ties are broken by the seeded hash.
    pending_weights = {gid: _group_features(cat, groups[gid]) for gid in pending}
    frequency: Counter[str] = Counter()
    for weights in pending_weights.values():
        frequency.update(weights.keys())
    pending.sort(
        key=lambda g: (
            min(frequency[f] for f in pending_weights[g]),
            hashlib.sha256(f"{config.seed}:{g}".encode()).hexdigest(),
        )
    )
    for gid in pending:
        weights = pending_weights[gid]
        if any(assigned["train"][f] == 0 for f in weights):
            split = "train"
        else:
            split = max(
                SPLITS,
                key=lambda s: (
                    sum(ratios[s] * (total[f] + w) - assigned[s][f] for f, w in weights.items()),
                    -SPLITS.index(s),
                ),
            )
        assigned[split].update(weights)
        total.update(weights)
        result[gid] = split
    support: Counter[str] = Counter()
    for gid, members in groups.items():
        if result[gid] != "heldout":
            for spec, label, _ in members:
                support.update([*_features(spec, cat.spec_factors), f"label={label}"])
    checked = {f for f, n in support.items() if n >= config.split_min_support}
    _refine(pending, pending_weights, result, assigned, total, ratios, checked)
    return result


_REFINE_PASSES = 50


def _refine(
    pending: Sequence[uuid.UUID],
    weights: Mapping[uuid.UUID, Counter],
    result: dict[uuid.UUID, str],
    assigned: dict[str, Counter],
    total: Counter,
    ratios: Mapping[str, float],
    checked: set[str],
) -> None:
    """Move new groups between splits while that lowers the imbalance.

    Imbalance is a chi-square over the features the split audit checks (values
    with at least split_min_support criteria): sum over splits of
    (assigned - ratio * total)^2 / total. Rarer values cannot be balanced and
    are left out, or they would dominate. A move never takes the last training
    weight of a value out of train. Deterministic: fixed order, best move per
    group, strict improvement only.
    """

    def cost(feature: str, split: str, value: float) -> float:
        expected = ratios[split] * total[feature]
        return (value - expected) ** 2 / total[feature]

    for _ in range(_REFINE_PASSES):
        moved = False
        for gid in pending:
            current = result[gid]
            w = weights[gid]
            best, best_delta = None, -1e-9
            for target in SPLITS:
                if target == current:
                    continue
                if current == "train" and any(assigned["train"][f] - n <= 0 for f, n in w.items()):
                    continue
                delta = sum(
                    cost(f, current, assigned[current][f] - n)
                    - cost(f, current, assigned[current][f])
                    + cost(f, target, assigned[target][f] + n)
                    - cost(f, target, assigned[target][f])
                    for f, n in w.items()
                    if f in checked
                )
                if delta < best_delta:
                    best, best_delta = target, delta
            if best is not None:
                assigned[current].subtract(w)
                assigned[best].update(w)
                result[gid] = best
                moved = True
        if not moved and not _swap_pass(pending, weights, result, assigned, cost, checked):
            break


def _swap_pass(pending, weights, result, assigned, cost, checked) -> bool:
    """Swap pairs of groups in different splits when that lowers the imbalance."""
    swapped = False
    for i, a in enumerate(pending):
        best, best_delta = None, -1e-9
        for b in pending[i + 1 :]:
            sa, sb = result[a], result[b]
            if sa == sb:
                continue
            wa, wb = weights[a], weights[b]
            change = {sa: Counter(), sb: Counter()}
            change[sa].update(wb)
            change[sa].subtract(wa)
            change[sb].update(wa)
            change[sb].subtract(wb)
            if any(
                s == "train" and n < 0 and assigned["train"][f] + n <= 0
                for s in (sa, sb)
                for f, n in change[s].items()
            ):
                continue
            delta = sum(
                cost(f, s, assigned[s][f] + n) - cost(f, s, assigned[s][f])
                for s in (sa, sb)
                for f, n in change[s].items()
                if n and f in checked
            )
            if delta < best_delta:
                best, best_delta = b, delta
        if best is not None:
            sa, sb = result[a], result[best]
            assigned[sa].subtract(weights[a])
            assigned[sa].update(weights[best])
            assigned[sb].subtract(weights[best])
            assigned[sb].update(weights[a])
            result[a], result[best] = sb, sa
            swapped = True
    return swapped


@dataclass(frozen=True)
class AuditRow:
    feature: str
    split: str
    share: float
    target: float
    support: int


@dataclass(frozen=True)
class SplitAudit:
    rows: list[AuditRow]
    failures: list[str]


def split_audit(
    cat: Catalogue,
    targets: Sequence[PlannedTarget],
    ratios: Mapping[str, float],
    *,
    tolerance: float = 0.02,
    min_support: int = 50,
) -> SplitAudit:
    """Share of each factor value and label per split, weighted by required count.

    Only features with at least `min_support` criteria are held to `tolerance`;
    smaller ones are reported but too small to balance.
    """
    failures: list[str] = []
    groups: dict[uuid.UUID, set[str]] = {}
    for t in targets:
        groups.setdefault(t.split_group_id, set()).add(t.split)
    failures += [f"split group {g} spans {sorted(s)}" for g, s in groups.items() if len(s) > 1]

    weights: dict[str, Counter] = {}
    support: Counter[str] = Counter()
    for t in targets:
        if t.split == "heldout":
            continue
        for feat in [*_features(t.spec, cat.spec_factors), f"label={t.intended_label}"]:
            weights.setdefault(feat, Counter())[t.split] += t.required_count
            support[feat] += 1
    rows = []
    for feat in sorted(weights):
        total = sum(weights[feat].values())
        for split in SPLITS:
            share = weights[feat][split] / total
            rows.append(AuditRow(feat, split, share, ratios[split], support[feat]))
            if support[feat] >= min_support and abs(share - ratios[split]) > tolerance + 1e-9:
                failures.append(f"{feat} {split}: {share:.3f} vs {ratios[split]:.3f}")
        if weights[feat]["train"] == 0:
            failures.append(f"{feat} has no training criteria")
    return SplitAudit(rows=rows, failures=failures)


# ---------------------------------------------------------------- build


def build_plan(cat: Catalogue, config: PlanConfig, existing: Sequence[ExistingTarget] = ()) -> Plan:
    """Deterministic criteria for `config`. Existing targets keep their split."""
    if any(lang not in cat.languages for lang in config.languages):
        raise ValueError(f"unknown language in {config.languages}")
    sampler = _Sampler(cat)
    existing_by_id = {e.combination_id: e for e in existing}
    non_stratified = {e.combination_id for e in existing if e.quota_group != "stratified"}

    rows: list[tuple[dict[str, str], str, int, str | None, int | None]] = []
    for language in config.languages:
        stratified = _stratified(cat, sampler, config, language, skip=non_stratified)
        rows += [(spec, "stratified", step, None, None) for step, spec in stratified]
        skip = {ids.combination_id(spec) for spec, *_ in rows}
        for priority, group_id, specs in _contrastive(cat, sampler, config, language, skip):
            rows += [
                (spec, "contrastive", priority, group_id, role) for role, spec in enumerate(specs)
            ]

    groups: dict[uuid.UUID, list[tuple[dict[str, str], str, int]]] = {}
    group_of: list[uuid.UUID] = []
    by_contrastive: dict[str, list[uuid.UUID]] = {}
    for spec, _, _, group_id, _ in rows:
        if group_id:
            by_contrastive.setdefault(group_id, []).append(ids.combination_id(spec))
    for spec, _, _, group_id, _ in rows:
        cid = ids.combination_id(spec)
        members = by_contrastive[group_id] if group_id else [cid]
        gid = (
            existing_by_id[cid].split_group_id
            if cid in existing_by_id
            else ids.split_group_id(members)
        )
        label = cat.intents[spec["caller_intent"]]["label"]
        count = (
            existing_by_id[cid].required_count if cid in existing_by_id else config.required_count
        )
        count = max(count, config.required_count)
        groups.setdefault(gid, []).append((spec, label, count))
        group_of.append(gid)

    known = {
        group_of[i]: existing_by_id[ids.combination_id(spec)].split
        for i, (spec, *_) in enumerate(rows)
        if ids.combination_id(spec) in existing_by_id
    }
    splits = _assign_splits(cat, config, groups, known)

    targets = []
    for (spec, quota_group, priority, group_id, role), gid in zip(rows, group_of, strict=True):
        cid = ids.combination_id(spec)
        derived = cat.derive(spec)
        count = existing_by_id[cid].required_count if cid in existing_by_id else 0
        targets.append(
            PlannedTarget(
                spec=spec,
                combination_id=cid,
                agent_state=derived["agent_state"],
                intended_label=derived["intended_label"],
                required_count=max(count, config.required_count),
                quota_group=quota_group,
                priority=priority,
                contrastive_group_id=group_id,
                contrastive_role=role,
                split_group_id=gid,
                split=splits[gid],
            )
        )
    return Plan(config=config, targets=targets)
