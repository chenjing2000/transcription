"""Lightweight segmentation rules; always preserve exact canonical spans."""
import pytest

from transcription.processing.text_structure import split_sentences


@pytest.mark.parametrize(("text", "expected"), [
    ("The value is 3.14. Next!", ["The value is 3.14.", "Next!"]),
    ("Dr. Smith arrived. Welcome!", ["Dr. Smith arrived.", "Welcome!"]),
    ("Mr. Jones said hello. Fine.", ["Mr. Jones said hello.", "Fine."]),
    ("The U.S. economy grew. Great!", ["The U.S. economy grew.", "Great!"]),
    ("A.B. testing works. Yes.", ["A.B. testing works.", "Yes."]),
    ("你好！他说：“真的？” 好。", ["你好！", "他说：“真的？”", "好。"]),
    ("One... Two!", ["One...", "Two!"]),
    ("No punctuation at end", ["No punctuation at end"]),
])
def test_sentence_boundaries_and_original_offsets(text, expected):
    sentences = split_sentences(text)
    assert [sentence.text for sentence in sentences] == expected
    assert all(text[s.char_start:s.char_end] == s.text for s in sentences)


def test_no_speech_only_whitespace():
    assert split_sentences("  \t \n ") == []
