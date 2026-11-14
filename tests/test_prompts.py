import json
import re

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation import prompts


@pytest.fixture(scope="module")
def cat():
    return load_catalogue()


SPEC = {
    "language": "hi",
    "regional_variant": "hi",
    "script_form": "romanized",
    "code_switching": "within_sentence",
    "code_switch_set": "en",
    "mixed_in_script": "default",
    "domain": "healthcare",
    "scenario": "healthcare_appointment",
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


def test_system_prompts_are_stable_and_complete(cat):
    system = prompts.render_generator_system(cat)
    assert system == prompts.render_generator_system(cat)
    assert "${" not in system
    for intent in cat.intents:
        assert f"`{intent}`" in system
    for activity in cat.values["agent_activity"]:
        assert f"`{activity}`" in system


def test_system_prompt_does_not_depend_on_the_criterion(cat):
    request = prompts.render_generation_request(cat, [SPEC], count=10)
    assert "agreement_token" in request
    assert "healthcare_appointment" in request
    assert "Hindi" in request and "English" in request
    assert "Administrative only" in request  # domain prompt notes
    assert "10 samples" in request
    assert "haan haan" in request  # Hindi hints


def test_hints_fall_back_to_english(cat):
    spec = {
        **SPEC,
        "caller_intent": "skip_request",
        "repetition": "single",
        "language": "en",
        "regional_variant": "en-US",
        "script_form": "native",
        "code_switching": "none",
        "code_switch_set": "none",
    }
    request = prompts.render_generation_request(cat, [spec], count=3)
    assert "skip that" in request


def test_contrastive_request_lists_members_and_shared_element(cat):
    group = next(g for g in cat.contrastive if g["id"] == "yes_backchannel_vs_answer")
    a = {**SPEC, **group["members"][0]}
    b = {**SPEC, **group["members"][1]}
    request = prompts.render_generation_request(cat, [a, b], count=4, group=group)
    assert "4 sets" in request
    assert "member_index 0" in request and "member_index 1" in request
    assert "exactly the same in every member" in request
    assert "8 samples" in request


def test_prompt_hash_changes_with_text():
    assert prompts.prompt_hash("a", "b") != prompts.prompt_hash("a", "c")


def _lang(spec, **kw):
    return {**spec, **kw}


def test_script_line_for_english_led_speech_romanizes_mixed_in_languages(cat):
    spec = _lang(
        SPEC, language="en", regional_variant="en-IN", script_form="native", code_switch_set="bn"
    )
    line = prompts.script_expectation(cat, spec)
    assert (
        line
        == "English in Latin letters; Bengali words romanized (Latin letters), never in Bengali-Assamese script."
    )


def test_script_line_for_native_hindi_with_english(cat):
    spec = _lang(SPEC, script_form="native", code_switch_set="en")
    assert (
        prompts.script_expectation(cat, spec)
        == "Hindi in Devanagari script; English words in Latin letters."
    )


def test_script_line_for_romanized_and_mixed_hindi(cat):
    assert prompts.script_expectation(cat, SPEC) == (
        "Hindi romanized (Latin letters), not Devanagari; English words in Latin letters."
    )
    mixed = _lang(SPEC, script_form="mixed_script", code_switching="none", code_switch_set="none")
    assert prompts.script_expectation(cat, mixed) == (
        "Hindi partly in Devanagari script and partly romanized (Latin letters) within the same utterance."
    )


def test_script_line_reaches_both_prompts(cat):
    spec = _lang(
        SPEC, language="en", regional_variant="en-IN", script_form="native", code_switch_set="bn"
    )
    expected = prompts.script_expectation(cat, spec)
    assert f"Script: {expected}" in prompts.render_generation_request(cat, [spec], count=2)
    item = prompts.JudgeItem(
        spec, "Guest", "hi", "Hotel Manager", "We have a suite on the", "Guest", "ok"
    )
    _, questions = prompts.render_judge_decision(cat, item)
    assert expected in questions["script_ok"]["instructions"]


def test_judge_decision_is_yes_no_about_the_named_people(cat):
    spec = {**SPEC, "cut_position": "middle"}
    item = prompts.JudgeItem(
        spec,
        "Guest",
        "Do you have anything with a view?",
        "Hotel Manager",
        "Sir, we have a suite on the highest floor, and",
        "Guest",
        "Mm-hmm, wait wait",
    )
    judge_input, questions = prompts.render_judge_decision(cat, item)
    assert json.loads(judge_input) == {
        "conversation": [
            {"speaker": "Guest", "transcript": "Do you have anything with a view?"},
            {
                "speaker": "Hotel Manager",
                "transcript": "Sir, we have a suite on the highest floor, and",
            },
            {"speaker": "Guest", "transcript": "Mm-hmm, wait wait"},
        ]
    }
    assert {k for k in questions if not k.startswith("factor_")} == {
        "is_backchannel",
        "is_barge_in",
        "prefix_1",
        "prefix_2",
        "language_ok",
        "script_ok",
        "cut_ok",
        "natural_ok",
    }
    assert all(q["type"] == "noul" for q in questions.values())
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) for key in questions)
    back = questions["is_backchannel"]["instructions"]
    assert (
        "While Hotel Manager is speaking, can the last turn in this conversation, said by Guest, be labelled a backchannel?"
        in back
    )
    assert '"Mm-hmm, wait"' in questions["prefix_2"]["instructions"]
    assert "break off" in questions["cut_ok"]["instructions"]
    text = judge_input + json.dumps(questions)
    assert "${" not in text
    assert "agreement_token" not in text and "intended" not in text  # blind
    for intent in cat.intents:
        if "_" in intent:  # ids such as agreement_token; plain words like "continuer" may appear
            assert intent not in text


def test_judge_question_after_the_speaker_finished(cat):
    spec = {**SPEC, "cut_position": "just_finished"}
    item = prompts.JudgeItem(
        spec, "Irina", "Is it ready?", "David", "Yes, it is ready.", "Irina", "ok"
    )
    _, questions = prompts.render_judge_decision(cat, item)
    assert (
        "Right after David finished speaking, can the last turn"
        in questions["is_barge_in"]["instructions"]
    )
    assert "complete" in questions["cut_ok"]["instructions"]
    assert not any(key.startswith("prefix_") for key in questions)  # one word: no proper prefixes


def test_script_line_for_malaysian_english_mixing_mandarin(cat):
    spec = _lang(
        SPEC, language="en", regional_variant="en-MY", script_form="native", code_switch_set="zh"
    )
    assert prompts.script_expectation(cat, {**spec, "mixed_in_script": "native"}) == (
        "English in Latin letters; Mandarin Chinese words in Simplified Chinese characters, not pinyin."
    )
    assert prompts.script_expectation(cat, {**spec, "mixed_in_script": "romanized"}) == (
        "English in Latin letters; Mandarin Chinese words in pinyin (Latin letters), "
        "never in Chinese characters."
    )


def test_script_line_for_hong_kong_english_mixing_cantonese(cat):
    spec = _lang(
        SPEC, language="en", regional_variant="en-HK", script_form="native", code_switch_set="yue"
    )
    assert "Cantonese words in Traditional Chinese characters" in prompts.script_expectation(
        cat, {**spec, "mixed_in_script": "native"}
    )
    assert "Cantonese words in Jyutping" in prompts.script_expectation(
        cat, {**spec, "mixed_in_script": "romanized"}
    )


def test_script_line_for_taiwan_mandarin(cat):
    spec = _lang(
        SPEC,
        language="zh",
        regional_variant="zh-TW",
        script_form="native",
        code_switching="none",
        code_switch_set="none",
    )
    assert (
        prompts.script_expectation(cat, spec)
        == "Mandarin Chinese in Traditional Chinese characters."
    )


def test_judge_confirms_every_generation_factor(cat):
    spec = {**SPEC, "cut_position": "middle"}
    item = prompts.JudgeItem(
        spec,
        "Patient",
        "I need to move my appointment.",
        "Receptionist",
        "The doctor is available on Monday at ten, and",
        "Patient",
        "haan haan",
    )
    _, questions = prompts.render_judge_decision(cat, item)
    factor_keys = {k for k in questions if k.startswith("factor_")}
    assert factor_keys == {
        "factor_scenario",
        "factor_speaker_activity",
        "factor_listener_intent",
        "factor_development",
        "factor_repetition",
        "factor_register",
        "factor_listener_mood",
    }
    assert "Booking or moving an appointment." in questions["factor_scenario"]["instructions"]
    assert "Receptionist" in questions["factor_speaker_activity"]["instructions"]
    assert (
        "Explaining a policy, process, status or reason."
        in questions["factor_speaker_activity"]["instructions"]
    )
    assert (
        "Agreement or affirmation that does not answer a question."
        in questions["factor_listener_intent"]["instructions"]
    )
    assert "repeated" in questions["factor_repetition"]["instructions"]
    assert "casual" in questions["factor_register"]["instructions"]
    assert "neutral" in questions["factor_listener_mood"]["instructions"]
