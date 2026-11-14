import json

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation.responses import JudgementError, parse_decision, parse_generation

CAT = load_catalogue()


SPEC = {
    "language": "en",
    "regional_variant": "en-US",
    "script_form": "native",
    "code_switching": "none",
    "code_switch_set": "none",
    "mixed_in_script": "default",
    "domain": "banking",
    "scenario": "banking_card_block",
    "speaker_pattern": "returning_speaker",
    "agent_activity": "explaining",
    "cut_position": "middle",
    "caller_intent": "correction",
    "development": "backchannel_then_barge_in",
    "repetition": "single",
    "length_bucket": "w4_8",
    "register": "casual",
    "caller_mood": "neutral",
}


def sample(**overrides):
    s = {
        "set_index": 0,
        "member_index": 0,
        "context_turn": {"speaker": "Customer", "text": "My card ending 1111 is lost."},
        "speaker": "Bank Officer",
        "speaker_full_text": "I have blocked the card ending 2222. A new one will arrive soon.",
        "speaker_spoken_text": "I have blocked the card ending",
        "listener": "Customer",
        "listener_text": "Yeah, but it ends in 1111",
        "label": "barge_in",
        "barge_in_from_word": 1,
        "rationale": "Starts as agreement, then corrects the card number.",
    }
    s.update(overrides)
    return s


def parse(samples, spec=SPEC, count=None):
    text = json.dumps({"samples": samples})
    return parse_generation(CAT, text, specs=[spec], count=count or len(samples))


def test_valid_sample_is_accepted():
    result = parse([sample()])
    assert result.fatal is None
    assert len(result.samples) == 1
    assert result.problems == []


def test_unparseable_response_is_fatal():
    result = parse_generation(CAT, "not json", specs=[SPEC], count=1)
    assert result.fatal is not None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"speaker_spoken_text": "I blocked"}, "spoken_not_prefix"),
        ({"speaker_spoken_text": "I have blocked the card end"}, "spoken_not_at_word_boundary"),
        (
            {
                "speaker_spoken_text": "I have blocked the card ending 2222. A new one will arrive soon."
            },
            "cut_position_mismatch",
        ),
        ({"listener_text": "No"}, "length_bucket_mismatch"),
        ({"barge_in_from_word": None}, "barge_in_word_missing"),
        ({"barge_in_from_word": 0}, "development_word_mismatch"),
        ({"barge_in_from_word": 9}, "barge_in_word_out_of_range"),
        ({"listener": "Bank Officer"}, "speaker_pattern_mismatch"),
        ({"context_turn": {"speaker": "Kim", "text": "Hi"}}, "speaker_pattern_mismatch"),
        ({"context_turn": {"speaker": "Bank Officer", "text": "Hi"}}, "speaker_pattern_mismatch"),
        ({"listener_text": "..."}, "empty_after_cleaning"),
    ],
)
def test_invalid_samples_are_rejected(overrides, reason):
    result = parse([sample(**overrides)])
    assert result.samples == []
    assert [p.reason for p in result.problems] == [reason]


def test_backchannel_must_not_have_a_barge_in_word():
    spec = {
        **SPEC,
        "caller_intent": "agreement_token",
        "development": "stable",
        "length_bucket": "w2_3",
    }
    result = parse(
        [sample(listener_text="yeah yeah", label="backchannel", barge_in_from_word=0)], spec
    )
    assert [p.reason for p in result.problems] == ["barge_in_word_not_allowed"]


def test_extra_samples_are_rejected():
    result = parse([sample(), sample()], count=1)
    assert len(result.samples) == 1
    assert [p.reason for p in result.problems] == ["extra_sample"]


def test_contrastive_sets_must_share_their_element():
    group = next(g for g in CAT.contrastive if g["id"] == "yes_backchannel_vs_answer")
    base = {
        **SPEC,
        "development": "stable",
        "length_bucket": "w1",
        "speaker_pattern": "returning_speaker",
    }
    a = {**base, **group["members"][0]}
    b = {**base, **group["members"][1]}
    full_q = "Shall I send the new card to your home address?"
    first = sample(listener_text="Yes", label="backchannel", barge_in_from_word=None)
    second = sample(
        member_index=1,
        speaker_full_text=full_q,
        speaker_spoken_text=full_q,
        listener_text="Yes",
        barge_in_from_word=0,
    )
    ok = parse_generation(
        CAT, json.dumps({"samples": [first, second]}), specs=[a, b], count=2, group=group
    )
    assert len(ok.samples) == 2, ok.problems
    second_bad = {**second, "listener_text": "Yeah"}
    bad = parse_generation(
        CAT, json.dumps({"samples": [first, second_bad]}), specs=[a, b], count=2, group=group
    )
    assert bad.samples == []
    assert {p.reason for p in bad.problems} == {"contrastive_shared_mismatch"}


def _answers(back=0.1, barge=0.9, prefixes=(0.2, 0.9, 0.95), **checks):
    p = {"is_backchannel": back, "is_barge_in": barge}
    p |= {f"prefix_{k}": v for k, v in enumerate(prefixes, start=1)}
    p |= {
        key: checks.get(key, 0.95) for key in ("language_ok", "script_ok", "cut_ok", "natural_ok")
    }
    return {key: {"type": "noul", "noul": v, "confidence": max(v, 1 - v)} for key, v in p.items()}


def test_parse_decision():
    verdict = parse_decision({"answers": _answers()})
    assert verdict.label == "barge_in"
    assert verdict.barge_in_from_word == 1  # prefix_2 is the first yes
    assert verdict.checks_failed == []
    assert "P(barge_in)=0.90" in verdict.notes


def test_decision_labels_from_two_yes_no_answers():
    assert parse_decision({"answers": _answers(back=0.9, barge=0.1)}).label == "backchannel"
    assert parse_decision({"answers": _answers(back=0.7, barge=0.8)}).label == "ambiguous"
    assert parse_decision({"answers": _answers(back=0.2, barge=0.3)}).label == "ambiguous"
    assert parse_decision({"answers": _answers(back=0.9, barge=0.1)}).barge_in_from_word is None


def test_barge_in_word_falls_back_to_the_full_turn():
    verdict = parse_decision({"answers": _answers(prefixes=(0.1, 0.2))})
    assert verdict.barge_in_from_word == 2  # only the full three-word turn is a barge-in
    assert parse_decision({"answers": _answers(prefixes=())}).barge_in_from_word == 0


def test_decision_checks_fail_below_one_half():
    verdict = parse_decision({"answers": _answers(script_ok=0.2)})
    assert verdict.checks_failed == ["script_ok"]


def test_blocked_decision_fails_every_check():
    verdict = parse_decision({"blocked": "safety"})
    assert verdict.checks_failed == ["blocked:safety"]


def test_incomplete_decision_is_an_error():
    answers = _answers()
    del answers["cut_ok"]
    with pytest.raises(JudgementError):
        parse_decision({"answers": answers})


def test_hyphenated_backchannel_is_one_word():
    spec = {**SPEC, "caller_intent": "continuer", "development": "stable", "length_bucket": "w1"}
    result = parse(
        [sample(listener_text="Mm-hmm.", label="backchannel", barge_in_from_word=None)], spec
    )
    assert result.problems == []


def test_barge_in_word_counts_original_words():
    # "Uh-huh," is one word, so "wait" is word 1 (cleaned text would make it word 2).
    spec = {**SPEC, "caller_intent": "floor_request", "development": "backchannel_then_barge_in"}
    result = parse([sample(listener_text="Uh-huh, wait wait hold on", barge_in_from_word=1)], spec)
    assert result.problems == []


@pytest.mark.parametrize(
    ("cut", "spoken", "ok"),
    [
        ("start", "I have", True),
        ("start", "I have blocked the card ending", False),
        ("early", "I have blocked", True),
        ("middle", "I have blocked the card ending", True),
        ("middle", "I have", False),
        ("near_end", "I have blocked the card ending 2222. A new one will", True),
        ("near_end", "I have blocked the card ending", False),
    ],
)
def test_cut_position_is_checked_from_the_spoken_share(cut, spoken, ok):
    spec = {**SPEC, "cut_position": cut}
    result = parse([sample(speaker_spoken_text=spoken)], spec)
    assert (result.problems == []) is ok, result.problems


def test_soft_checks_are_recorded_but_do_not_reject():
    answers = _answers(natural_ok=0.2)
    answers["factor_repetition"] = {"type": "noul", "noul": 0.1, "confidence": 0.9}
    answers["factor_listener_intent"] = {"type": "noul", "noul": 0.9, "confidence": 0.9}
    verdict = parse_decision(
        {"answers": answers},
        hard_checks=("language_ok", "script_ok", "cut_ok", "factor_listener_intent"),
    )
    assert verdict.checks_failed == []
    assert verdict.soft == {"natural_ok": 0.2, "factor_repetition": 0.1}
    assert "soft: P(natural_ok)=0.20, P(factor_repetition)=0.10" in verdict.notes


def test_hard_factor_check_rejects():
    answers = _answers()
    answers["factor_listener_intent"] = {"type": "noul", "noul": 0.3, "confidence": 0.7}
    verdict = parse_decision({"answers": answers}, hard_checks=("cut_ok", "factor_listener_intent"))
    assert verdict.checks_failed == ["factor_listener_intent"]


@pytest.mark.parametrize("mark", [" —", "-", " –"])  # ellipses: ellipsis_used
def test_the_cut_is_never_marked(mark):
    spoken = "I have blocked the card ending" + mark
    full = spoken + " 2222. A new one will arrive soon."
    result = parse([sample(speaker_spoken_text=spoken, speaker_full_text=full)])
    assert [p.reason for p in result.problems] == ["cut_marked"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"listener_text": "Umm... yeah, but it ends in 1111"},
        {"context_turn": {"speaker": "Customer", "text": "My card… it is lost."}},
        {"speaker_full_text": "I have blocked the card ending 2222... A new one will arrive soon."},
    ],
)
def test_ellipses_are_rejected_in_any_line(overrides):
    result = parse([sample(**overrides)])
    assert [p.reason for p in result.problems] == ["ellipsis_used"]
