"""Utterance criteria: per-language prompt lines and the judge's yes/no questions."""

import json

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation import prompts
from barge_in_detector.generation.utterance_criteria import (
    JudgementError,
    judge_questions,
    language_parts,
    parse_verdict,
)

CAT = load_catalogue()


def spec(**overrides):
    base = {
        "language": "en",
        "regional_variant": "en-US",
        "script_form": "native",
        "code_switching": "none",
        "code_switch_set": "none",
        "mixed_in_script": "default",
        "caller_intent": "acknowledgment",
        "development": "stable",
        "repetition": "single",
        "length_bucket": "w2_3",
    }
    return {**base, **overrides}


URDU = spec(
    language="ur",
    regional_variant="ur-IN",
    code_switching="within_sentence",
    code_switch_set="en",
    caller_intent="floor_request",
    development="filler_then_barge_in",
    repetition="repeated",
    length_bucket="w9_plus",
)
HINDI_MIX = spec(
    language="hi",
    regional_variant="hi",
    script_form="native",
    code_switching="within_sentence",
    code_switch_set="en",
    caller_intent="agreement_token",
    length_bucket="w4_8",
)
INDIAN_ENGLISH_HINDI = spec(
    language="en",
    regional_variant="en-IN",
    code_switching="within_sentence",
    code_switch_set="hi",
    mixed_in_script="native",
    caller_intent="correction",
    development="backchannel_then_barge_in",
    length_bucket="w9_plus",
)


def request(s):
    return prompts.render_utterance_request(CAT, s, count=5)


# ---------------------------------------------------------------- prompt


def test_every_language_in_the_combination_gets_its_own_script_line():
    text = request(URDU)
    assert "every language listed here must appear in every utterance" in text
    urdu = next(line for line in text.splitlines() if line.strip().startswith("- Urdu"))
    assert "main" not in text and "more than half" not in text  # no language ranks first
    assert "Perso-Arabic" in urdu
    assert "never in Devanagari" in urdu
    english = next(line for line in text.splitlines() if line.strip().startswith("- English"))
    assert "Latin letters" in english


def test_indian_words_in_indian_english_are_in_their_native_script():
    text = request(INDIAN_ENGLISH_HINDI)
    hindi = next(line for line in text.splitlines() if line.strip().startswith("- Hindi"))
    assert "in Devanagari script, never in Latin letters" in hindi
    assert not CAT.violations(INDIAN_ENGLISH_HINDI, partial_spec=True)
    romanized = {**INDIAN_ENGLISH_HINDI, "mixed_in_script": "romanized"}
    assert CAT.violations(romanized, partial_spec=True)
    # The prompt still knows how to ask for romanized words, should a set allow it.
    hindi = next(line for line in request(romanized).splitlines() if "- Hindi" in line)
    assert "as people type it in messages" in hindi
    # User decisions, 2026-11-15: Mandarin in characters only, no pinyin;
    # Indian languages in their native script only (alone or with English).
    for code in ("zh", "hi", "kn", "ta", "te", "ml", "ur"):
        assert CAT.languages[code]["romanized_usage"] == "none"
        options = CAT.language_options(code)
        assert {o[1] for o in options} == {"native"}


def test_every_language_of_the_combination_is_listed():
    parts = language_parts(CAT, HINDI_MIX)
    assert [p.code for p in parts] == ["hi", "en"]
    text = request(HINDI_MIX)
    for name in ("Hindi", "English"):
        assert any(line.strip().startswith(f"- {name}") for line in text.splitlines())


def test_request_keeps_the_factor_lines():
    text = request(URDU)
    for line in ("- Development: `filler_then_barge_in`", "- Length: `w9_plus`"):
        assert line in text
    assert "(intended label: barge_in)" in text


# ---------------------------------------------------------------- judge questions


def keys(s, text="Umm, wait wait, hold on a second please, I need to check.", word=1):
    judge_input, questions = judge_questions(CAT, s, text, word)
    assert json.loads(judge_input) == {"text": text}
    for q in questions.values():
        assert q["type"] == "noul" and q["instructions"]
    return set(questions)


def test_one_question_per_criterion_and_one_per_language():
    assert keys(HINDI_MIX, "Haan bilkul, सही, exactly.", None) == {
        "language_hi",
        "language_en",
        "script_hi",
        "script_en",
        "switching",
        "intent",
        "development",
        "repetition",
    }


def test_a_regional_variant_is_a_criterion_of_its_own():
    assert "variant" in keys(URDU)
    assert "variant" not in keys(HINDI_MIX, "Haan bilkul, सही, exactly.", None)


def test_judge_questions_are_plain_one_line_questions():
    _, questions = judge_questions(CAT, URDU, "ہاں ہاں، hold on", 2)
    assert questions["script_ur"]["instructions"] == (
        "Are the Urdu words in this text written in Urdu (Perso-Arabic) script?"
    )
    assert questions["language_en"]["instructions"] == "Does this text contain English words?"
    _, questions = judge_questions(CAT, INDIAN_ENGLISH_HINDI, "हाँ हाँ, but not this one", 2)
    assert questions["script_hi"]["instructions"] == (
        "Are the Hindi words in this text written in Devanagari script?"
    )
    assert questions["language_hi"]["instructions"] == "Does this text contain Hindi words?"
    _, questions = judge_questions(CAT, URDU, "ہاں ہاں، hold on", 2)
    assert questions["language_ur"]["instructions"] == "Does this text contain Urdu words?"


def test_development_question_names_the_turning_word():
    s = spec(
        caller_intent="correction",
        development="backchannel_then_barge_in",
        repetition="repeated",
        length_bucket="w9_plus",
    )
    _, questions = judge_questions(CAT, s, "Sure sure, but that is not my email address.", 2)
    assert questions["development"]["instructions"] == (
        'Does this text start as a backchannel and become an interruption at the word "but"?'
    )
    _, questions = judge_questions(CAT, spec(), "Got it.", None)
    assert questions["development"]["instructions"] == (
        "Does this text keep the same intent from start to end?"
    )


# ---------------------------------------------------------------- verdicts


def answers(**p):
    return {"answers": {k: {"type": "noul", "noul": v} for k, v in p.items()}}


def test_every_question_answered_yes_accepts():
    verdict = parse_verdict(answers(language_en=0.9, script_en=0.8), ["language_en", "script_en"])
    assert verdict.failed == []
    assert verdict.answers == {"language_en": 0.9, "script_en": 0.8}


def test_a_question_answered_no_fails_by_its_key():
    verdict = parse_verdict(
        answers(language_ar=0.2, language_fr=0.9, script_ar=0.49),
        ["language_ar", "language_fr", "script_ar"],
    )
    assert verdict.failed == ["language_ar", "script_ar"]


def test_blocked_input_fails_every_question():
    verdict = parse_verdict({"blocked": "safety"}, ["intent"])
    assert verdict.failed == ["blocked:safety"]


def test_missing_answer_is_unusable():
    with pytest.raises(JudgementError):
        parse_verdict(answers(intent=0.9), ["intent", "repetition"])


def test_request_spells_out_switching_majority_and_repetition():
    between = spec(
        language="id",
        regional_variant="id",
        code_switching="between_sentences",
        code_switch_set="en",
        caller_intent="floor_request",
        development="filler_then_barge_in",
        repetition="repeated",
        length_bucket="w9_plus",
    )
    text = request(between)
    assert "each language in its own sentence, ended with a full stop" in text
    assert "- Repetition: `repeated`: say a key word or short phrase two or more times" in text
    assert "- Repetition: `single`: no word or phrase is said twice" in request(spec())
