"""Factor catalogue: loading, spec validation and derived values.

The catalogue is versioned data under generation/factors/ and is explained in
docs/factor-catalogue.md.
"""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "generation" / "factors"

# Native scripts written without spaces between words: word counts do not apply.
UNSPACED_SCRIPTS = {"Thai", "Laoo", "Khmr", "Mymr", "Jpan", "Hans", "Hant"}

# Factors that describe the conversation's language; fixed together per spec.
LANGUAGE_FACTORS = (
    "language",
    "regional_variant",
    "script_form",
    "code_switching",
    "code_switch_set",
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class Catalogue:
    root: Path
    version: str
    file_versions: dict[str, str]
    spec_factors: list[str]
    values: dict[str, list[str]]
    descriptions: dict[tuple[str, str], str]
    factor_descriptions: dict[str, str]
    languages: dict[str, dict[str, Any]]
    domains: dict[str, dict[str, Any]]
    scenarios: dict[str, dict[str, Any]]
    intents: dict[str, dict[str, Any]]
    rules: list[dict[str, Any]]
    contrastive: list[dict[str, Any]]
    labels: list[str] = field(default_factory=list)

    # ---- language helpers ----
    def variants(self, language: str) -> list[str]:
        lang = self.languages[language]
        return list(lang["regional_variants"] or [language])

    def code_switch_sets(self, language: str, variant: str) -> list[str]:
        sets = self.languages[language]["code_switch_sets"]
        return [
            "+".join(s["languages"])
            for s in sets
            if variant in s.get("regional_variants", [variant])
        ]

    def language_options(self, language: str) -> list[tuple[str, str, str, str]]:
        """Valid (regional_variant, script_form, code_switching, code_switch_set)."""
        options = []
        for variant in self.variants(language):
            sets = ["none", *self.code_switch_sets(language, variant)]
            for script, switching, cs_set in itertools.product(
                self.values["script_form"], self.values["code_switching"], sets
            ):
                partial = {
                    "language": language,
                    "regional_variant": variant,
                    "script_form": script,
                    "code_switching": switching,
                    "code_switch_set": cs_set,
                }
                if switching != "none" and cs_set == "none":
                    continue
                if not self.violations(partial, partial_spec=True):
                    options.append((variant, script, switching, cs_set))
        return options

    # ---- validation ----
    def _attribute(self, spec: Mapping[str, str], key: str) -> str | None:
        language = spec.get("language")
        if language not in self.languages:
            return None
        attr = key.removeprefix("language.")
        if attr == "code_switch_sets":
            variant = spec.get("regional_variant", language)
            return "nonempty" if self.code_switch_sets(language, variant) else "empty"
        return str(self.languages[language][attr])

    def _value(self, spec: Mapping[str, str], key: str) -> str | None:
        if key.startswith("language."):
            return self._attribute(spec, key)
        return spec.get(key)

    def violations(self, spec: Mapping[str, str], *, partial_spec: bool = False) -> list[str]:
        """Names of the rules a spec breaks; empty when valid.

        With partial_spec, absent factors are ignored rather than reported.
        """
        out: list[str] = []
        for factor in self.spec_factors:
            if factor not in spec:
                if not partial_spec:
                    out.append(f"missing:{factor}")
            elif spec[factor] not in self.values[factor]:
                out.append(f"unknown_value:{factor}")
        if out and not partial_spec:
            return out

        language = spec.get("language")
        if language in self.languages:
            variant = spec.get("regional_variant")
            if variant is not None and variant not in self.variants(language):
                out.append("variant_in_language")
            cs_set = spec.get("code_switch_set")
            if spec.get("code_switching", "none") != "none" and cs_set == "none":
                out.append("code_switch_set_needed")
            if (
                variant is not None
                and cs_set not in (None, "none")
                and cs_set not in self.code_switch_sets(language, variant)
            ):
                out.append("code_switch_set_in_language")
        scenario = spec.get("scenario")
        if (
            scenario in self.scenarios
            and "domain" in spec
            and self.scenarios[scenario]["domain"] != spec["domain"]
        ):
            out.append("scenario_in_domain")
        intent = self.intents.get(spec.get("caller_intent", ""))
        if intent is not None:
            if "length_bucket" in spec and spec["length_bucket"] not in intent["length_buckets"]:
                out.append("intent_length_bucket")
            if spec.get("repetition") == "repeated" and not intent["repeatable"]:
                out.append("intent_repeatable")

        for rule in self.rules:
            when = rule["when"]
            current = {key: self._value(spec, key) for key in when}
            if any(v is None for v in current.values()):
                continue
            if not all(current[key] in [str(x) for x in allowed] for key, allowed in when.items()):
                continue
            broken = any(
                key in spec and spec[key] not in [str(x) for x in allowed]
                for key, allowed in rule.get("require", {}).items()
            ) or any(
                key in spec and spec[key] in [str(x) for x in disallowed]
                for key, disallowed in rule.get("forbid", {}).items()
            )
            if broken:
                out.append(rule["id"])
        return out

    def is_valid(self, spec: Mapping[str, str]) -> bool:
        return not self.violations(spec)

    def derive(self, spec: Mapping[str, str]) -> dict[str, str]:
        state = "just_finished" if spec["cut_position"] == "just_finished" else "speaking"
        return {
            "agent_state": state,
            "intended_label": self.intents[spec["caller_intent"]]["label"],
        }

    def allowed_values(self, factor: str, fixed: Mapping[str, str]) -> list[str]:
        """Values of `factor` that do not break a rule given only `fixed`."""
        return [
            v
            for v in self.values[factor]
            if not self.violations({**fixed, factor: v}, partial_spec=True)
        ]

    # ---- reference checks ----
    def check(self) -> list[str]:
        """Reference errors in the catalogue files; empty when consistent."""
        errors: list[str] = []
        codes = set(self.languages)
        for code, lang in self.languages.items():
            for s in lang["code_switch_sets"]:
                for other in s["languages"]:
                    if other not in codes:
                        errors.append(f"{code}: unknown code-switch language {other}")
                    if other == code:
                        errors.append(f"{code}: mixes itself")
                for v in s.get("regional_variants", []):
                    if v not in self.variants(code):
                        errors.append(f"{code}: set limited to unknown variant {v}")
        for intent_id, intent in self.intents.items():
            if intent["label"] not in self.labels:
                errors.append(f"intent {intent_id}: unknown label {intent['label']}")
            for bucket in intent["length_buckets"]:
                if bucket not in self.values["length_bucket"]:
                    errors.append(f"intent {intent_id}: unknown length bucket {bucket}")
        attrs = {
            "language.romanized_usage": ["none", "occasional", "common"],
            "language.code_switch_sets": ["empty", "nonempty"],
        }

        def check_refs(where: str, mapping: Mapping[str, Any]) -> None:
            for key, vs in mapping.items():
                allowed = self.values.get(key) or attrs.get(key)
                if allowed is None:
                    errors.append(f"{where}: unknown factor {key}")
                    continue
                for v in vs if isinstance(vs, list) else [vs]:
                    if str(v) not in allowed:
                        errors.append(f"{where}: {key}={v} is not a value")

        for rule in self.rules:
            for part in ("when", "require", "forbid"):
                check_refs(rule["id"], rule.get(part, {}))
        for group in self.contrastive:
            for member in group["members"]:
                check_refs(group["id"], member)
                if not self._satisfiable(member):
                    errors.append(f"contrastive {group['id']}: member {member} is unsatisfiable")
        for intent_id in self.intents:
            if not self._satisfiable({"caller_intent": intent_id}):
                errors.append(f"intent {intent_id} has no valid combination")
        return errors

    CORE_FACTORS = (
        "agent_activity",
        "cut_position",
        "caller_intent",
        "development",
        "repetition",
        "length_bucket",
        "caller_mood",
        "speaker_pattern",
    )

    def _satisfiable(self, fixed: Mapping[str, str]) -> bool:
        free = [f for f in self.CORE_FACTORS if f not in fixed]
        for combo in itertools.product(*(self.allowed_values(f, fixed) for f in free)):
            if not self.violations(
                {**fixed, **dict(zip(free, combo, strict=True))}, partial_spec=True
            ):
                return True
        return False


def load_catalogue(root: Path = DEFAULT_ROOT) -> Catalogue:
    root = Path(root)
    cat = _load(root / "catalogue.yaml")
    files = {name: _load(root / filename) for name, filename in cat["files"].items()}
    languages = {str(lang["code"]): lang for lang in files["languages"]["languages"]}
    domains = {d["id"]: d for d in files["domains"]["domains"]}
    scenarios = {
        s["id"]: {"domain": d["id"], "description": s["description"]}
        for d in files["domains"]["domains"]
        for s in d["scenarios"]
    }
    intents = {i["id"]: i for i in files["intents"]["intents"]}

    spec_factors = [f["id"] for f in cat["spec_factors"]]
    values: dict[str, list[str]] = {}
    descriptions: dict[tuple[str, str], str] = {}
    factor_descriptions: dict[str, str] = {}
    for f in cat["spec_factors"]:
        factor_descriptions[f["id"]] = " ".join(str(f.get("description", "")).split())
        for v in f.get("values", []):
            values[f["id"]] = [*values.get(f["id"], []), str(v["id"])]
            if "description" in v:
                descriptions[(f["id"], str(v["id"]))] = " ".join(str(v["description"]).split())
    values["language"] = list(languages)
    values["regional_variant"] = sorted(
        {v for code, lang in languages.items() for v in (lang["regional_variants"] or [code])}
    )
    values["code_switch_set"] = [
        "none",
        *sorted(
            {
                "+".join(s["languages"])
                for lang in languages.values()
                for s in lang["code_switch_sets"]
            }
        ),
    ]
    values["domain"] = list(domains)
    values["scenario"] = list(scenarios)
    values["caller_intent"] = list(intents)
    labels = next(d["values"] for d in cat["derived_factors"] if d["id"] == "intended_label")

    return Catalogue(
        root=root,
        version=str(cat["version"]),
        file_versions={"catalogue": str(cat["version"])}
        | {name: str(data["version"]) for name, data in files.items()},
        spec_factors=spec_factors,
        values=values,
        descriptions=descriptions,
        factor_descriptions=factor_descriptions,
        languages=languages,
        domains=domains,
        scenarios=scenarios,
        intents=intents,
        rules=files["constraints"]["rules"],
        contrastive=files["contrastive"]["groups"],
        labels=list(labels),
    )
