import uuid

from barge_in_detector import ids

NAMESPACE = uuid.UUID("3de823b9-152c-4f71-899d-8d2bde9cc65c")


def test_namespace_is_the_documented_one():
    assert ids.NAMESPACE == NAMESPACE


def test_combination_id_matches_documented_derivation():
    spec = {"language": "hi", "caller_intent": "continuer", "cut_position": "middle"}
    expected = uuid.uuid5(
        NAMESPACE, "criterion:caller_intent=continuer;cut_position=middle;language=hi"
    )
    assert ids.combination_id(spec) == expected


def test_combination_id_ignores_key_order():
    a = {"language": "hi", "caller_intent": "continuer"}
    b = {"caller_intent": "continuer", "language": "hi"}
    assert ids.combination_id(a) == ids.combination_id(b)


def test_split_group_id_uses_smallest_combination_id():
    a = uuid.UUID("00000000-0000-5000-8000-000000000002")
    b = uuid.UUID("00000000-0000-5000-8000-000000000001")
    expected = uuid.uuid5(NAMESPACE, f"split_group:{b}")
    assert ids.split_group_id([a, b]) == expected


def test_example_id_matches_documented_derivation():
    conversation_id = uuid.UUID("019a8a2e-0000-7000-8000-000000000001")
    expected = uuid.uuid5(NAMESPACE, f"example:{conversation_id}:3:2")
    assert ids.example_id(conversation_id, turn_shape=3, prefix_index=2) == expected


def test_new_id_is_time_ordered_v7():
    first, second = ids.new_id(), ids.new_id()
    assert first.version == 7
    assert first < second


def test_factors_at_their_default_value_are_left_out_of_the_id():
    spec = {"language": "en", "caller_intent": "continuer"}
    assert ids.combination_id({**spec, "mixed_in_script": "default"}) == ids.combination_id(spec)
    assert ids.combination_id({**spec, "mixed_in_script": "native"}) != ids.combination_id(spec)
