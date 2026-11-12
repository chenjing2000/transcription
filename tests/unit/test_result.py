import pytest
from transcription.result import Word


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
