import json

import pytest

from barge_in_detector.catalogue import load_catalogue
from barge_in_detector.generation.responses import parse_generation, parse_judgement

CAT = load_catalogue()


SPEC = {
    "language": "en",
    "regional_variant": "en-US",
    "script_form": "native",
    "code_switching": "none",
    "code_switch_set": "none",
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
        "context_turn": {
            "speaker": "Sam",
            "role": "caller",
            "text": "My card ending 1111 is lost.",
        },
        "agent_speaker": "Agent",
        "agent_full_text": "I have blocked the card ending 2222. A new one will arrive soon.",
        "agent_spoken_text": "I have blocked the card ending",
        "candidate_speaker": "Sam",
        "candidate_role": "caller",
        "candidate_text": "Yeah, but it ends in 1111",
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
        ({"agent_spoken_text": "I blocked"}, "agent_spoken_not_prefix"),
        ({"agent_spoken_text": "I have blocked the card end"}, "agent_spoken_not_at_word_boundary"),
        (
            {
                "agent_spoken_text": "I have blocked the card ending 2222. A new one will arrive soon."
            },
            "cut_position_mismatch",
        ),
        ({"candidate_text": "No"}, "length_bucket_mismatch"),
        ({"barge_in_from_word": None}, "barge_in_word_missing"),
        ({"barge_in_from_word": 0}, "development_word_mismatch"),
        ({"barge_in_from_word": 9}, "barge_in_word_out_of_range"),
        ({"candidate_speaker": "Agent"}, "speaker_pattern_mismatch"),
        (
            {"context_turn": {"speaker": "Kim", "role": "caller", "text": "Hi"}},
            "speaker_pattern_mismatch",
        ),
        ({"candidate_text": "..."}, "empty_after_cleaning"),
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
        [sample(candidate_text="yeah yeah", label="backchannel", barge_in_from_word=0)], spec
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
    first = sample(candidate_text="Yes", label="backchannel", barge_in_from_word=None)
    second = sample(
        member_index=1,
        agent_full_text=full_q,
        agent_spoken_text=full_q,
        candidate_text="Yes",
        barge_in_from_word=0,
    )
    ok = parse_generation(
        CAT, json.dumps({"samples": [first, second]}), specs=[a, b], count=2, group=group
    )
    assert len(ok.samples) == 2, ok.problems
    second_bad = {**second, "candidate_text": "Yeah"}
    bad = parse_generation(
        CAT, json.dumps({"samples": [first, second_bad]}), specs=[a, b], count=2, group=group
    )
    assert bad.samples == []
    assert {p.reason for p in bad.problems} == {"contrastive_shared_mismatch"}


def test_parse_judgement():
    text = json.dumps(
        {
            "items": [
                {
                    "index": 0,
                    "label": "barge_in",
                    "barge_in_from_word": 1,
                    "language_ok": True,
                    "script_ok": True,
                    "cut_ok": True,
                    "natural_ok": True,
                    "notes": "",
                }
            ]
        }
    )
    verdicts = parse_judgement(text, count=1)
    assert verdicts[0].label == "barge_in"
    assert verdicts[0].checks_failed == []
