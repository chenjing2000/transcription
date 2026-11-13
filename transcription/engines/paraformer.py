"""FunASR Paraformer adapter: lazy VAD/ASR/punctuation and conservative timing."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from transcription.engines.base import BaseEngine
from transcription.errors import TranscriptionError
from transcription.result import Character, EngineInfo, Paragraph, Sentence, TranscriptionResult, Word

ENGINE_SPECS = (EngineInfo(
    name="paraformer_zh", model_id="paraformer-zh", language="zh",
    description="FunASR Paraformer with FSMN-VAD and CT punctuation",
),)

HAN = re.compile(r"[\u3400-\u9fff]")
LEXEME = re.compile(r"[\u3400-\u9fff]+|[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\s]")


def is_han(value: str) -> bool:
    return bool(value) and all(HAN.fullmatch(c) for c in value)


def visible_units(text: str) -> list[tuple[str, int, int]]:
    items: list[tuple[str, int, int]] = []
    for match in LEXEME.finditer(text):
        token = match.group()
        if is_han(token):
            items.extend((c, match.start() + i, match.start() + i + 1)
                         for i, c in enumerate(token))
        elif token[0].isalnum():
            items.append((token, match.start(), match.end()))
    return items


def raw_units(raw: object) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    result: list[str] = []
    for unit in raw.split():
        result.extend(unit if is_han(unit) else [unit])
    return result


def intervals_ms(raw: object) -> list[tuple[float, float]] | None:
    if not isinstance(raw, (list, tuple)):
        return None
    result = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            return None
        start, end = pair[:2]
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float)) or not 0 <= start <= end):
            return None
        result.append((float(start) / 1000, float(end) / 1000))
    return result


def word_spans(text: str) -> list[tuple[str, int, int]]:
    try:
        import jieba
    except ImportError:
        jieba = None
    words: list[tuple[str, int, int]] = []
    for match in LEXEME.finditer(text):
        token = match.group()
        if is_han(token):
            parts = list(jieba.cut(token, HMM=False)) if jieba else list(token)
            index = match.start()
            for part in parts:
                if not part:
                    continue
                words.append((part, index, index + len(part)))
                index += len(part)
        elif token[0].isalnum():
            words.append((token, match.start(), match.end()))
    return words


def normalize_paraformer_result(record: dict[str, Any]) -> tuple[str, list[Word], list[Character], list[str]]:
    text = str(record.get("text") or "").strip()
    if not text:
        return "", [], [], []
    units = visible_units(text)
    raw = raw_units(record.get("raw_text"))
    intervals = intervals_ms(record.get("timestamp"))
    verified = bool(raw and intervals is not None and len(raw) == len(intervals)
                    and raw == [token for token, _, _ in units])
    warnings: list[str] = []
    if not verified:
        warnings.append("Character timestamps unverified: raw token alignment missing or mismatched")
    indexed: dict[int, tuple[float, float]] = {}
    token_times: dict[tuple[int, int], tuple[float, float]] = {}
    characters: list[Character] = []
    if verified:
        for (token, begin, end), pair in zip(units, intervals):
            token_times[(begin, end)] = pair
            if len(token) == 1 and is_han(token):
                indexed[begin] = pair
                characters.append(Character(token, begin, pair[0], pair[1], "aligned"))
    else:
        characters = [Character(text[i], i) for i, c in enumerate(text) if HAN.fullmatch(c)]
    words: list[Word] = []
    for token, begin, end in word_spans(text):
        pair = None
        if verified:
            if is_han(token) and all(i in indexed for i in range(begin, end)):
                pair = (indexed[begin][0], indexed[end - 1][1])
            else:
                pair = token_times.get((begin, end))
        words.append(Word(
            text=token, start=pair[0] if pair else None,
            end=pair[1] if pair else None, char_start=begin, char_end=end,
            alignment_status="aligned" if pair else "missing",
        ))
    if any(item.start is None for item in words):
        warnings.append("Some words lack verified time intervals")
    return text, words, characters, warnings


class ParaformerEngine(BaseEngine):
    def __init__(self, *, model_id: str, name: str, device: str):
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise TranscriptionError("DEPENDENCY_MISSING", "FunASR is not installed") from exc
        self.name = name
        self.device = device
        self.pipeline = None
        try:
            self.pipeline = AutoModel(
                model=model_id, device=device,
                vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000},
                punc_model="ct-punc", disable_update=True,
            )
        except Exception as exc:
            raise TranscriptionError("MODEL_LOAD_FAILED", f"Paraformer model loading failed: {exc}") from exc

    def transcribe(self, file_path: Path) -> TranscriptionResult:
        try:
            output = self.pipeline.generate(
                input=str(file_path), batch_size_s=60,
                batch_size_threshold_s=30, return_raw_text=True,
                sentence_timestamp=True, output_timestamp=True,
            )
        except Exception as exc:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"Paraformer recognition failed: {exc}") from exc
        if not isinstance(output, list) or not output or not isinstance(output[0], dict):
            raise TranscriptionError("TRANSCRIPTION_FAILED", "FunASR returned no structured result")
        text, words, characters, warnings = normalize_paraformer_result(output[0])
        if not text:
            return TranscriptionResult(
                status="partial", info="No speech detected", engine=self.name,
                source_path=file_path, device=self.device, warnings=["Empty speech result"],
            )
        sentence = Sentence(text, 0, len(text))
        return TranscriptionResult(
            status="partial", info="Chinese recognition complete; file export not yet implemented",
            engine=self.name, source_path=file_path, device=self.device,
            text=text, words=words, characters=characters,
            sentences=[sentence], paragraphs=[Paragraph(text, [sentence])],
            warnings=warnings,
        )

    def close(self) -> None:
        self.pipeline = None
