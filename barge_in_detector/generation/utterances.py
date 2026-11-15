"""Utterance track: the factor set and the seeded plan builder.

One criterion is a combination of the factors in
generation/factors/utterances.yaml; Gemini writes `required_count` caller
utterances for it. See docs/utterance-first-decision.md.

The plan is built position by position, so it grows prefix-stable: a larger
`criteria_per_language` keeps every earlier criterion and its split.
"""

import dataclasses
import itertools
import random
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from barge_in_detector import ids
from barge_in_detector.catalogue import LANGUAGE_FACTORS, Catalogue

TRACK_FILE = "utterances.yaml"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class UtteranceTrack:
    version: str
    factors: tuple[str, ...]
    excluded_intents: dict[str, str]
    intents: tuple[str, ...]  # catalogue order, excluded ones removed
    labels: tuple[str, ...] = ("backchannel", "barge_in")
    rules: tuple[dict[str, Any], ...] = ()  # added to the catalogue's constraints

    def catalogue(self, cat: Catalogue) -> Catalogue:
        """The catalogue with this track's rules added, for validating its specs."""
        return dataclasses.replace(cat, rules=[*cat.rules, *self.rules])


def load_utterance_track(cat: Catalogue) -> UtteranceTrack:
    data = yaml.safe_load((cat.root / TRACK_FILE).read_text(encoding="utf-8"))
    excluded = dict(data["excluded_intents"])
    unknown = set(excluded) - set(cat.intents)
    if unknown:
        raise ValueError(f"unknown excluded intents: {sorted(unknown)}")
    factors = tuple(data["factors"])
    if set(factors) - set(cat.spec_factors):
        raise ValueError(f"unknown factors: {sorted(set(factors) - set(cat.spec_factors))}")
    labels = tuple(data["labels"])
    intents = tuple(i for i in cat.intents if i not in excluded)
    stray = {i: cat.intents[i]["label"] for i in intents if cat.intents[i]["label"] not in labels}
    if stray:
        raise ValueError(f"intents with labels outside {labels}: {stray}")
    return UtteranceTrack(
        version=str(data["version"]),
        factors=factors,
        excluded_intents=excluded,
        intents=intents,
        labels=labels,
        rules=tuple(data.get("rules") or ()),
    )


@dataclass(frozen=True)
class UtterancePlanConfig:
    name: str
    seed: int
    languages: list[str]
    criteria_per_language: int
    required_count: int
    label_shares: dict[str, float]
    # Share of criteria per regional variant; languages not listed spread evenly.
    variant_shares: dict[str, dict[str, float]] = field(default_factory=dict)
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    # Candidates drawn per position; the one using the least-covered factor
    # values wins.
    candidates_per_pick: int = 8
    # Every valid combination of each language instead of criteria_per_language
    # picks, each with the same required_count.
    all_combinations: bool = False
    # Keep only these values of a factor, e.g. {"code_switching": ["none"]}.
    only: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> UtterancePlanConfig:
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UtteranceTarget:
    spec: dict[str, str]
    combination_id: uuid.UUID
    intended_label: str
    required_count: int
    priority: int
    split: str


def _next_by_deficit(shares: Mapping[str, float], counts: Counter) -> str:
    """The key furthest below its share so far; ties go to the first key."""
    total = sum(counts.values()) + 1
    return max(shares, key=lambda k: (shares[k] * total - counts[k], -list(shares).index(k)))


def split_for(combination_id: uuid.UUID, ratios: Mapping[str, float]) -> str:
    """A deterministic split from the ID alone, in the given proportions."""
    point = (combination_id.int % 10_000) / 10_000
    edge = 0.0
    for split in SPLITS:
        edge += ratios.get(split, 0.0)
        if point < edge:
            return split
    return SPLITS[0]


class _Builder:
    def __init__(self, cat: Catalogue, track: UtteranceTrack, config: UtterancePlanConfig):
        self.cat = track.catalogue(cat)
        self.track = track
        self.config = config
        self._options: dict[str, list[dict[str, str]]] = {}
        self._rest: dict[str, list[dict[str, str]]] = {}
        self._valid: dict[tuple[str, str, str], list[dict[str, str]]] = {}

    def language_specs(self, language: str) -> list[dict[str, str]]:
        if language not in self._options:
            self._options[language] = [
                dict(zip(LANGUAGE_FACTORS, (language, *option), strict=True))
                for option in self.cat.language_options(language)
            ]
        return self._options[language]

    def intent_specs(self, intent: str) -> list[dict[str, str]]:
        """Valid values of the track's factors other than language and intent."""
        if intent not in self._rest:
            rest = [
                f for f in self.track.factors if f not in LANGUAGE_FACTORS and f != "caller_intent"
            ]
            self._rest[intent] = [
                spec
                for values in itertools.product(*(self.cat.values[f] for f in rest))
                if not self.cat.violations(
                    spec := {"caller_intent": intent, **dict(zip(rest, values, strict=True))},
                    partial_spec=True,
                )
            ]
        return self._rest[intent]

    def valid_specs(self, language: str, variant: str, intent: str) -> list[dict[str, str]]:
        """Every valid spec for a variant and intent, in a fixed order."""
        key = (language, variant, intent)
        if key not in self._valid:
            self._valid[key] = [
                spec
                for side in self.language_specs(language)
                if side["regional_variant"] == variant
                for rest in self.intent_specs(intent)
                if not self.cat.violations(spec := {**side, **rest}, partial_spec=True)
                and all(spec[f] in values for f, values in self.config.only.items())
            ]
        return self._valid[key]

    def variant_shares(self, language: str) -> dict[str, float]:
        shares = self.config.variant_shares.get(language)
        if shares:
            return dict(shares)
        variants = self.cat.variants(language)
        return {v: 1 / len(variants) for v in variants}

    def build_all(self, language: str) -> list[UtteranceTarget]:
        """Every valid combination of a language, ordered by combination_id."""
        specs = {
            ids.utterance_combination_id(spec): spec
            for variant in dict.fromkeys(
                s["regional_variant"] for s in self.language_specs(language)
            )
            for intent in self.track.intents
            for spec in self.valid_specs(language, variant, intent)
        }
        return [
            UtteranceTarget(
                spec={f: spec[f] for f in self.track.factors},
                combination_id=cid,
                intended_label=self.cat.intents[spec["caller_intent"]]["label"],
                required_count=self.config.required_count,
                priority=position,
                split=split_for(cid, self.config.split_ratios),
            )
            for position, (cid, spec) in enumerate(sorted(specs.items()))
        ]

    def build(self, language: str) -> list[UtteranceTarget]:
        config = self.config
        label_counts: Counter = Counter()
        variant_counts: Counter = Counter()
        intent_turn: Counter = Counter()
        coverage: Counter = Counter()
        chosen: set[uuid.UUID] = set()
        by_label = {
            label: [i for i in self.track.intents if self.cat.intents[i]["label"] == label]
            for label in config.label_shares
        }
        targets = []
        for position in range(config.criteria_per_language):
            label = _next_by_deficit(config.label_shares, label_counts)
            label_counts[label] += 1
            intents = by_label[label]
            if not intents:
                continue
            intent = intents[intent_turn[label] % len(intents)]
            intent_turn[label] += 1
            variant = _next_by_deficit(self.variant_shares(language), variant_counts)
            variant_counts[variant] += 1
            rng = random.Random(f"{config.seed}:{language}:{position}")
            valid = [
                (spec, cid)
                for spec in self.valid_specs(language, variant, intent)
                if (cid := ids.utterance_combination_id(spec)) not in chosen
            ]
            candidates = []
            for index in sorted(
                rng.sample(range(len(valid)), min(len(valid), config.candidates_per_pick))
            ):
                spec, cid = valid[index]
                score = sum(coverage[(f, spec[f])] for f in self.track.factors)
                candidates.append((score, index, spec, cid))
            if not candidates:
                continue
            _, _, spec, cid = min(candidates)
            chosen.add(cid)
            coverage.update((f, spec[f]) for f in self.track.factors)
            targets.append(
                UtteranceTarget(
                    spec={f: spec[f] for f in self.track.factors},
                    combination_id=cid,
                    intended_label=self.cat.intents[intent]["label"],
                    required_count=config.required_count,
                    priority=position,
                    split=split_for(cid, config.split_ratios),
                )
            )
        return targets


def build_utterance_plan(
    cat: Catalogue, track: UtteranceTrack, config: UtterancePlanConfig
) -> list[UtteranceTarget]:
    """Criteria for every language in the config, in generation order. Specs obey
    the catalogue's constraints and the track's own rules."""
    unknown = set(config.label_shares) - set(track.labels)
    if unknown:
        raise ValueError(f"label_shares has labels outside {track.labels}: {sorted(unknown)}")
    builder = _Builder(cat, track, config)
    plan: list[UtteranceTarget] = []
    for language in config.languages:
        plan += builder.build_all(language) if config.all_combinations else builder.build(language)
    return plan


def spec_of(row: Mapping[str, Any], factors: Sequence[str]) -> dict[str, str]:
    return {f: row[f] for f in factors}


def plan_targets(cat: Catalogue, plan: Sequence[UtteranceTarget]) -> list[dict[str, Any]]:
    """The plan as the runner and the store use it: one dict per criterion with
    its factor values, id, intended label, required count, split and locales.
    The plan file is the only record of criteria (it is versioned in git)."""
    return [
        {
            **t.spec,
            "combination_id": t.combination_id,
            "label": t.intended_label,
            "required_count": t.required_count,
            "split": t.split,
            "locales": cat.locales(t.spec),
        }
        for t in plan
    ]
