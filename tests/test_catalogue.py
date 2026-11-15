from pathlib import Path

import pytest

from barge_in_detector.catalogue import load_catalogue

ROOT = Path(__file__).resolve().parents[1] / "generation" / "factors"


@pytest.fixture(scope="module")
def cat():
    return load_catalogue(ROOT)


def base_spec(**overrides):
    spec = {
        "language": "hi",
        "regional_variant": "hi",
        "script_form": "romanized",
        "code_switching": "within_sentence",
        "code_switch_set": "en",
        "mixed_in_script": "default",
        "domain": "banking",
        "scenario": "banking_card_block",
        "speaker_pattern": "returning_speaker",
        "agent_activity": "explaining",
        "cut_position": "middle",
        "caller_intent": "agreement_token",
        "development": "stable",
        "repetition": "repeated",
        "length_bucket": "w2_3",
        "register": "casual",
        "caller_mood": "neutral",
    }
    spec.update(overrides)
    return spec


def test_shipped_catalogue_has_no_reference_errors(cat):
    assert cat.check() == []


def test_valid_spec_has_no_violations(cat):
    assert cat.violations(base_spec()) == []


def test_answer_needs_a_finished_question(cat):
    assert "answer_needs_finished_question" in cat.violations(
        base_spec(caller_intent="answer_to_question", repetition="single")
    )
    assert (
        cat.violations(
            base_spec(
                caller_intent="answer_to_question",
                agent_activity="asking_yes_no_question",
                cut_position="just_finished",
            )
        )
        == []
    )


def test_code_switching_needs_a_set(cat):
    assert "code_switch_set_needed" in cat.violations(base_spec(code_switch_set="none"))


def test_language_attribute_rules(cat):
    # English has romanized_usage none.
    spec = base_spec(
        language="en",
        regional_variant="en-US",
        script_form="romanized",
        code_switching="none",
        code_switch_set="none",
    )
    assert "no_romanized_without_usage" in cat.violations(spec)


def test_membership_is_checked(cat):
    assert "scenario_in_domain" in cat.violations(base_spec(scenario="travel_baggage"))
    assert "variant_in_language" in cat.violations(base_spec(regional_variant="en-IN"))
    # Hindi mixing for English is limited to en-IN.
    spec = base_spec(
        language="en",
        regional_variant="en-US",
        script_form="native",
        code_switch_set="hi",
    )
    assert "code_switch_set_in_language" in cat.violations(spec)


def test_intent_length_and_repetition(cat):
    assert "intent_length_bucket" in cat.violations(base_spec(length_bucket="w9_plus"))
    assert "intent_repeatable" in cat.violations(
        base_spec(caller_intent="tracking_repeat", agent_activity="reading_numbers")
    )


def test_unknown_value_is_a_violation(cat):
    assert "unknown_value:caller_mood" in cat.violations(base_spec(caller_mood="sleepy"))


def test_derived_values(cat):
    assert cat.derive(base_spec()) == {"agent_state": "speaking", "intended_label": "backchannel"}
    finished = base_spec(
        caller_intent="answer_to_question",
        agent_activity="asking_yes_no_question",
        cut_position="just_finished",
    )
    assert cat.derive(finished) == {"agent_state": "just_finished", "intended_label": "barge_in"}


def test_language_options_follow_the_manifest(cat):
    options = cat.language_options("en")
    assert ("en-US", "native", "none", "none", "default") in options
    assert ("en-IN", "native", "within_sentence", "kn", "default") in options
    assert not any(o[1] == "romanized" for o in options)
    assert not any(o[0] == "en-US" and o[2] != "none" for o in options)


def test_locales_list_every_language_in_the_transcript(cat):
    plain_us = base_spec(
        language="en",
        regional_variant="en-US",
        script_form="native",
        code_switching="none",
        code_switch_set="none",
    )
    assert cat.locales(plain_us) == ["en-US"]
    indian = {
        **plain_us,
        "regional_variant": "en-IN",
        "code_switching": "within_sentence",
        "code_switch_set": "kn",
    }
    assert cat.locales(indian) == ["en-IN", "kn-Latn-IN"]  # romanized in English-led speech
    hindi = base_spec()  # romanized Hindi (region IN) mixing English
    assert cat.locales(hindi) == ["hi-Latn-IN", "en-IN"]
    assert cat.locales({**hindi, "script_form": "native"}) == ["hi-IN", "en-IN"]


def test_mandarin_english_code_switching_in_malaysia(cat):
    assert "zh" in cat.code_switch_sets("en", "en-MY")
    assert "zh" not in cat.code_switch_sets("en", "en-US")
    assert "en" in cat.code_switch_sets("zh", "zh-MY")


def _english(**kw):
    return base_spec(language="en", script_form="native", **kw)


def test_mixed_in_chinese_comes_in_characters_and_romanized(cat):
    options = cat.language_options("en")
    my = {o for o in options if o[0] == "en-MY" and o[3] == "zh"}
    assert {o[4] for o in my} == {"native", "romanized"}
    hk = {o for o in options if o[0] == "en-HK" and o[3] == "yue"}
    assert {o[4] for o in hk} == {"native", "romanized"}
    # Indian sets keep the default (romanized) only.
    assert {o[4] for o in options if o[0] == "en-IN"} == {"default"}
    assert {o[4] for o in options if o[2] == "none"} == {"default"}


def test_mixed_in_script_must_fit_the_set(cat):
    ok = _english(
        regional_variant="en-MY",
        code_switching="within_sentence",
        code_switch_set="zh",
        mixed_in_script="romanized",
    )
    assert cat.violations(ok) == []
    assert "mixed_in_script_in_set" in cat.violations({**ok, "mixed_in_script": "default"})
    indian = _english(
        regional_variant="en-IN",
        code_switching="within_sentence",
        code_switch_set="hi",
        mixed_in_script="native",
    )
    assert "mixed_in_script_in_set" in cat.violations(indian)


def test_locales_carry_script_subtags_for_chinese(cat):
    base = _english(
        regional_variant="en-MY", code_switching="within_sentence", code_switch_set="zh"
    )
    assert cat.locales({**base, "mixed_in_script": "native"}) == ["en-MY", "zh-Hans-MY"]
    assert cat.locales({**base, "mixed_in_script": "romanized"}) == ["en-MY", "zh-Latn-MY"]
    hk = _english(
        regional_variant="en-HK",
        code_switching="within_sentence",
        code_switch_set="yue",
        mixed_in_script="native",
    )
    assert cat.locales(hk) == ["en-HK", "yue-Hant-HK"]
    taiwan = base_spec(
        language="zh",
        regional_variant="zh-TW",
        script_form="native",
        code_switching="none",
        code_switch_set="none",
    )
    assert cat.locales(taiwan) == ["zh-Hant-TW"]


def test_indian_languages_mix_only_with_english(cat):
    """User decision, 2026-11-15: an Indian language alone or with English;
    Indian English with one Indian language. No three-language sets."""
    for code in ("hi", "kn", "ta", "te", "ml", "bn", "gu", "mr", "pa", "as", "ur"):
        for variant in cat.languages[code].get("regional_variants") or [code]:
            assert set(cat.code_switch_sets(code, variant)) <= {"en"}, (code, variant)
    for other in cat.code_switch_sets("en", "en-IN"):
        assert "+" not in other
