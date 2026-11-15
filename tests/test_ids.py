import uuid

from barge_in_detector import ids

NAMESPACE = uuid.UUID("3de823b9-152c-4f71-899d-8d2bde9cc65c")


def test_namespace_is_the_documented_one():
    assert ids.NAMESPACE == NAMESPACE


def test_combination_id_matches_documented_derivation():
    spec = {"language": "hi", "caller_intent": "continuer", "length_bucket": "w1"}
    expected = uuid.uuid5(
        NAMESPACE, "utterance_criterion:caller_intent=continuer;language=hi;length_bucket=w1"
    )
    assert ids.utterance_combination_id(spec) == expected


def test_combination_id_ignores_key_order():
    a = {"language": "hi", "caller_intent": "continuer"}
    b = {"caller_intent": "continuer", "language": "hi"}
    assert ids.utterance_combination_id(a) == ids.utterance_combination_id(b)


def test_new_id_is_time_ordered_v7():
    first, second = ids.new_id(), ids.new_id()
    assert first.version == 7
    assert first < second


def test_factors_at_their_default_value_are_left_out_of_the_id():
    spec = {"language": "en", "caller_intent": "continuer"}
    with_default = {**spec, "mixed_in_script": "default"}
    assert ids.utterance_combination_id(with_default) == ids.utterance_combination_id(spec)
    native = {**spec, "mixed_in_script": "native"}
    assert ids.utterance_combination_id(native) != ids.utterance_combination_id(spec)
