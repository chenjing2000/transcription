import pytest
from transcription.result import Word, Character


def test_valid_word():
    word = Word("hello", 0.0, 0.5, 1, 6, "aligned")
    assert word.end == 0.5


@pytest.mark.parametrize("args", [
    ("x", 1.0, None, 0, 1, "missing"),
    ("x", 2.0, 1.0, 0, 1, "aligned"),
    ("x", None, None, 2, 1, "missing"),
    ("x", None, None, 0, 1, "aligned"),
])
def test_bad_word(args):
    with pytest.raises(ValueError):
        Word(*args)


def test_word_rejects_nonfinite_timing():
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            Word("x", value, value, 0, 1, "aligned")


def test_character_timestamp_invariants():
    for args in [
        ("你", -1, None, None, "missing"),
        ("你好", 0, None, None, "missing"),
        ("你", 0, 0.2, None, "missing"),
        ("你", 0, float("inf"), 1.0, "aligned"),
        ("你", 0, None, None, "aligned"),
    ]:
        with pytest.raises(ValueError):
            Character(*args)
