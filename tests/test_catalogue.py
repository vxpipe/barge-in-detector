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
    assert ("en-US", "native", "none", "none") in options
    assert ("en-IN", "native", "within_sentence", "hi+kn") in options
    assert not any(o[1] == "romanized" for o in options)
    assert not any(o[0] == "en-US" and o[2] != "none" for o in options)
