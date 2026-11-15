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
        "script_form": "native",
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
    assert ("en-IN", "native", "within_sentence", "kn", "native") in options
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
    assert cat.locales({**indian, "mixed_in_script": "native"}) == ["en-IN", "kn-IN"]
    hindi = base_spec()  # native Hindi (region IN) mixing English
    assert cat.locales(hindi) == ["hi-IN", "en-IN"]


def test_english_varieties_and_mandarin_in_singapore(cat):
    # User decision, 2026-11-15: English as spoken in the US, India and Singapore.
    assert cat.languages["en"]["regional_variants"] == ["en-US", "en-IN", "en-SG"]
    assert cat.languages["ta"]["regional_variants"] == ["ta-IN"]
    assert cat.languages["ur"]["regional_variants"] == ["ur-IN"]
    assert cat.languages["es"]["regional_variants"] == ["es-ES", "es-MX"]
    assert cat.languages["fr"]["regional_variants"] == ["fr-FR"]
    assert cat.languages["de"]["regional_variants"] == ["de-DE"]
    assert cat.languages["zh"]["regional_variants"] == ["zh-CN"]
    assert "zh" in cat.code_switch_sets("en", "en-SG")
    assert "zh" not in cat.code_switch_sets("en", "en-US")
    assert "en" in cat.code_switch_sets("zh", "zh-CN")


def _english(**kw):
    return base_spec(language="en", script_form="native", **kw)


def test_mixed_in_chinese_comes_in_characters_and_romanized(cat):
    options = cat.language_options("en")
    sg = {o for o in options if o[0] == "en-SG" and o[3] == "zh"}
    # User decision, 2026-11-15: no pinyin anywhere; characters only.
    assert {o[4] for o in sg} == {"native"}
    # User decision, 2026-11-15: Indian words in Indian English are written in
    # their own native script, never romanized.
    assert {o[4] for o in options if o[0] == "en-IN" and o[2] != "none"} == {"native"}
    assert {o[4] for o in options if o[2] == "none"} == {"default"}


def test_mixed_in_script_must_fit_the_set(cat):
    ok = _english(
        regional_variant="en-SG",
        code_switching="within_sentence",
        code_switch_set="zh",
        mixed_in_script="native",
    )
    assert cat.violations(ok) == []
    assert "mixed_in_script_in_set" in cat.violations({**ok, "mixed_in_script": "romanized"})
    assert "mixed_in_script_in_set" in cat.violations({**ok, "mixed_in_script": "default"})
    indian = _english(
        regional_variant="en-IN",
        code_switching="within_sentence",
        code_switch_set="hi",
        mixed_in_script="native",
    )
    assert cat.violations(indian) == []
    assert "mixed_in_script_in_set" in cat.violations({**indian, "mixed_in_script": "romanized"})


def test_locales_carry_script_subtags_for_chinese(cat):
    base = _english(
        regional_variant="en-SG", code_switching="within_sentence", code_switch_set="zh"
    )
    assert cat.locales({**base, "mixed_in_script": "native"}) == ["en-SG", "zh-Hans-SG"]
    mainland = base_spec(
        language="zh",
        regional_variant="zh-CN",
        script_form="native",
        code_switching="none",
        code_switch_set="none",
    )
    assert cat.locales(mainland) == ["zh-Hans-CN"]


def test_indian_languages_mix_only_with_english(cat):
    """User decision, 2026-11-15: an Indian language alone or with English;
    Indian English with one Indian language. No three-language sets."""
    for code in ("hi", "kn", "ta", "te", "ml", "ur"):
        for variant in cat.languages[code].get("regional_variants") or [code]:
            assert set(cat.code_switch_sets(code, variant)) <= {"en"}, (code, variant)
    for other in cat.code_switch_sets("en", "en-IN"):
        assert "+" not in other
