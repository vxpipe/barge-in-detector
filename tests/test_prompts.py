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
    judge = prompts.render_judge_system(cat)
    assert "${" not in judge
    # The judge is blind: no intent ids.
    assert "agreement_token" not in judge


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
