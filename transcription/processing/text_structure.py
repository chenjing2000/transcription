"""Sentence and paragraph segmentation with original-text character spans."""
from __future__ import annotations

import re

from transcription.errors import TranscriptionError
from transcription.result import Paragraph, Sentence, TranscriptionResult

_END = re.compile(r'[.!?。！？]+(?:[”’"\u201d\u2019])?(?=\s|$|[^.!?。！？])')


def split_sentences(text: str) -> list[Sentence]:
    """Split at terminal punctuation without rewriting the source string."""
    if not text.strip():
        return []
    sentences: list[Sentence] = []
    pos = 0
    for match in _END.finditer(text):
        end = match.end()
        # Include trailing closing punctuation, but not leading whitespace.
        while pos < end and text[pos].isspace():
            pos += 1
        if pos < end:
            sentences.append(Sentence(text[pos:end], pos, end))
        pos = end
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text):
        sentences.append(Sentence(text[pos:], pos, len(text)))
    return sentences


def build_paragraphs(text: str, sentences: list[Sentence],
                     *, max_chars: int = 420, max_sentences: int = 3) -> list[Paragraph]:
    """Group contiguous sentences. No speculative semantic rephrasing."""
    if not sentences:
        return []
    groups: list[list[Sentence]] = []
    current: list[Sentence] = []
    for sentence in sentences:
        if current and (len(current) >= max_sentences or
                        sentence.char_end - current[0].char_start > max_chars):
            groups.append(current)
            current = []
        current.append(sentence)
    if current:
        groups.append(current)
    return [Paragraph(text[g[0].char_start:g[-1].char_end], g) for g in groups]


def structure_result(result: TranscriptionResult) -> None:
    """Validate aligned offsets and derive sentences and paragraphs in-place."""
    text = result.text
    if not text or not text.strip():
        raise TranscriptionError("NO_SPEECH", "Recognition produced no speech text")
    for word in result.words:
        if (word.char_start < 0 or word.char_end > len(text) or
                word.char_start >= word.char_end or
                text[word.char_start:word.char_end] != word.text):
            raise TranscriptionError("INVALID_ALIGNMENT", f"Word character span mismatch: {word.text!r}")
    if result.characters is not None:
        for char in result.characters:
            if (char.char_index < 0 or char.char_index >= len(text)
                    or text[char.char_index] != char.text):
                raise TranscriptionError("INVALID_ALIGNMENT", "Character position mismatch")
    result.sentences = split_sentences(text)
    result.paragraphs = build_paragraphs(text, result.sentences)
    if not result.words and not any("word" in x.lower() for x in result.warnings):
        result.warnings.append("No verified word alignment entries")
    elif any(w.start is None for w in result.words) and not any("timestamp" in x.lower() for x in result.warnings):
        result.warnings.append("Some word timestamps are unavailable")
