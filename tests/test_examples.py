import uuid

from barge_in_detector import ids
from barge_in_detector.examples import BUILDER_VERSION, build_examples
from barge_in_detector.storage.schema import SPEC_FACTORS

CONVERSATION = {
    "conversation_id": uuid.UUID("019a8a2e-0000-7000-8000-000000000001"),
    "run_id": uuid.UUID("019a8a2e-0000-7000-8000-0000000000ff"),
    "combination_id": uuid.UUID("00000000-0000-5000-8000-000000000001"),
    "factor_catalogue_version": "0.7.0",
    **{f: "x" for f in SPEC_FACTORS},
    "language": "hi",
    "locales": ["hi-IN", "en-IN"],
    "speaker_pattern": "returning_speaker",
    "agent_state": "speaking",
    "label": "barge_in",
    "contrastive_group_id": None,
    "split_group_id": uuid.UUID("00000000-0000-5000-8000-000000000002"),
    "split": "train",
    "turns": [{"speaker": "Priya", "role": "caller", "text": "Haan, theek hai; lekin..."}],
    "agent_speaker": "Agent",
    "agent_spoken_text": "Aapka order kal tak pahunch jayega, aur",
    "candidate_speaker": "Priya",
    "candidate_role": "caller",
    "candidate_text": "Haan haan, but — change the time!",
    "barge_in_from_word": 2,
}


def test_builds_both_shapes_and_every_prefix():
    rows = build_examples([CONVERSATION])
    # 7 original tokens; "—" cleans to nothing new, so 6 prefixes x 2 turn shapes.
    assert len(rows) == 12
    assert {r["turn_shape"] for r in rows} == {2, 3}
    assert all(r["builder_version"] == BUILDER_VERSION for r in rows)


def test_prefix_labels_follow_the_barge_in_word():
    rows = [r for r in build_examples([CONVERSATION]) if r["turn_shape"] == 3]
    rows.sort(key=lambda r: r["prefix_index"])
    assert [(r["prefix_index"], r["candidate_clean"], r["label"], r["label_id"]) for r in rows] == [
        (0, "haan", "backchannel", 0),
        (1, "haan haan", "backchannel", 0),
        (2, "haan haan but", "barge_in", 1),
        (4, "haan haan but change", "barge_in", 1),
        (5, "haan haan but change the", "barge_in", 1),
        (6, "haan haan but change the time", "barge_in", 1),
    ]


def test_windows_and_ids():
    rows = {(r["turn_shape"], r["prefix_index"]): r for r in build_examples([CONVERSATION])}
    final3 = rows[(3, 6)]
    assert final3["is_final_prefix"]
    assert final3["transcript_clean"] == (
        "s1: haan theek hai lekin\ns2: aapka order kal tak pahunch jayega aur\n"
        "s1: haan haan but change the time"
    )
    assert final3["speaker_labels"] == ["s1", "s2", "s1"]
    assert final3["transcript_original"].endswith("Priya: Haan haan, but — change the time!")
    assert final3["example_id"] == ids.example_id(
        CONVERSATION["conversation_id"], turn_shape=3, prefix_index=6
    )
    assert final3["transcript_raw"] == (
        "s1: Haan, theek hai; lekin...\ns2: Aapka order kal tak pahunch jayega, aur\n"
        "s1: Haan haan, but — change the time!"
    )
    final2 = rows[(2, 6)]
    assert final2["transcript_clean"] == (
        "s1: aapka order kal tak pahunch jayega aur\ns2: haan haan but change the time"
    )


def test_backchannel_and_ambiguous_conversations():
    back = {
        **CONVERSATION,
        "label": "backchannel",
        "barge_in_from_word": None,
        "candidate_text": "haan haan",
    }
    assert {r["label"] for r in build_examples([back])} == {"backchannel"}
    amb = {**back, "label": "ambiguous"}
    rows = build_examples([amb])
    assert {r["label"] for r in rows} == {"ambiguous"}
    assert {r["label_id"] for r in rows} == {None}


def test_output_is_deterministic():
    other = {**CONVERSATION, "conversation_id": uuid.UUID("019a8a2e-0000-7000-8000-000000000002")}
    assert build_examples([CONVERSATION, other]) == build_examples([other, CONVERSATION])


def test_hyphenated_word_before_the_barge_in_point():
    conv = {**CONVERSATION, "candidate_text": "Uh-huh, wait a second", "barge_in_from_word": 1}
    rows = sorted(
        (r for r in build_examples([conv]) if r["turn_shape"] == 2), key=lambda r: r["prefix_index"]
    )
    assert [(r["candidate_clean"], r["label"]) for r in rows] == [
        ("uh huh", "backchannel"),
        ("uh huh wait", "barge_in"),
        ("uh huh wait a", "barge_in"),
        ("uh huh wait a second", "barge_in"),
    ]
