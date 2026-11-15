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
    "mixed_in_script",
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

    def code_switch_set_entry(self, language: str, variant: str, cs_set: str) -> dict[str, Any]:
        """The manifest entry for a code-switch set ("en+hi") of a variant."""
        for s in self.languages[language]["code_switch_sets"]:
            if "+".join(s["languages"]) == cs_set and variant in s.get(
                "regional_variants", [variant]
            ):
                return s
        raise KeyError(f"{language}/{variant} has no code-switch set {cs_set}")

    def region(self, language: str, variant: str) -> str | None:
        if "-" in variant:
            return variant.split("-", 1)[1]
        return self.languages[language].get("region")

    def native_script(self, language: str, region: str | None) -> str:
        """The native script of a language in a region (zh-TW: Hant)."""
        lang = self.languages[language]
        variant_scripts = lang.get("variant_scripts") or {}
        return variant_scripts.get(f"{language}-{region}", lang["scripts"][0])

    def latin(self, language: str) -> bool:
        return self.languages[language]["scripts"][0] == "Latn"

    def mixed_in_scripts(self, spec: Mapping[str, str]) -> list[str]:
        """Allowed mixed_in_script values for a spec's language side."""
        if spec.get("code_switching", "none") == "none" or spec.get("code_switch_set") in (
            None,
            "none",
        ):
            return ["default"]
        try:
            entry = self.code_switch_set_entry(
                spec["language"], spec["regional_variant"], spec["code_switch_set"]
            )
        except KeyError:
            return ["default"]
        return list(entry.get("mixed_in_scripts") or ["default"])

    def mixed_in_script_codes(self, spec: Mapping[str, str]) -> dict[str, str]:
        """The script (ISO 15924, or "mixed") each mixed-in language is written in."""
        language, form = spec["language"], spec["script_form"]
        region = self.region(language, spec["regional_variant"])
        choice = spec.get("mixed_in_script", "default")
        latin_led = form == "romanized" or (form == "native" and self.latin(language))
        codes: dict[str, str] = {}
        if spec.get("code_switching", "none") == "none" or spec.get("code_switch_set") in (
            None,
            "none",
        ):
            return codes
        for other in spec["code_switch_set"].split("+"):
            if self.latin(other) or choice == "romanized" or (choice == "default" and latin_led):
                codes[other] = "Latn"
            elif choice == "native":
                codes[other] = self.native_script(other, region)
            elif form == "native":
                codes[other] = self.native_script(language, region)
            else:
                codes[other] = "mixed"
        return codes

    def locales(self, spec: Mapping[str, str]) -> list[str]:
        """Every locale in the transcript, the conversation's first, then each
        mixed-in language's in the same region. A script subtag is added when
        a language is not written in its usual script, or has several native
        scripts: ["en-US"], ["en-MY", "zh-Hans-MY"], ["en-MY", "zh-Latn-MY"],
        ["hi-Latn-IN", "en-IN"], ["zh-Hant-TW"]."""
        language, variant = spec["language"], spec["regional_variant"]
        region = self.region(language, variant)

        def tag(code: str, script: str | None) -> str:
            usual = self.native_script(code, region)
            # Chinese tags conventionally always name the script (zh-Hans, yue-Hant).
            several = len(self.languages[code]["scripts"]) > 1 or usual in ("Hans", "Hant")
            parts = [code]
            if script and script != "mixed" and (script != usual or several):
                parts.append(script)
            if region:
                parts.append(region)
            return "-".join(parts)

        form = spec.get("script_form", "native")
        main_script = {"romanized": "Latn", "native": self.native_script(language, region)}.get(
            form
        )
        locales = [tag(language, main_script)]
        for other, script in self.mixed_in_script_codes(spec).items():
            locales.append(tag(other, None if script == "Latn" and self.latin(other) else script))
        return locales

    def language_options(self, language: str) -> list[tuple[str, str, str, str, str]]:
        """Valid (regional_variant, script_form, code_switching, code_switch_set,
        mixed_in_script)."""
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
                for mixed in self.mixed_in_scripts(partial):
                    candidate = {**partial, "mixed_in_script": mixed}
                    if not self.violations(candidate, partial_spec=True):
                        options.append((variant, script, switching, cs_set, mixed))
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
            mixed = spec.get("mixed_in_script")
            if (
                mixed is not None
                and variant is not None
                and cs_set is not None
                and mixed not in self.mixed_in_scripts(spec)
            ):
                out.append("mixed_in_script_in_set")
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
                allowed = s.get("mixed_in_scripts")
                if allowed is not None and (
                    not set(allowed) <= {"native", "romanized"}
                    or any(o in self.languages and self.latin(o) for o in s["languages"])
                ):
                    errors.append(
                        f"{code}: mixed_in_scripts {allowed} needs native/romanized "
                        "and non-Latin languages"
                    )
        for intent_id, intent in self.intents.items():
            if intent["label"] not in self.labels:
                errors.append(f"intent {intent_id}: unknown label {intent['label']}")
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
        for intent_id in self.intents:
            if not self._satisfiable({"caller_intent": intent_id}):
                errors.append(f"intent {intent_id} has no valid combination")
        return errors

    CORE_FACTORS = (
        "agent_activity",
        "cut_position",
        "caller_intent",
        "development",
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
        labels=list(labels),
    )
