# Paraformer Chinese ASR — Phase 5

The `paraformer_zh` engine uses FunASR `AutoModel` with `model="paraformer-zh"`, `vad_model="fsmn-vad"` and `punc_model="ct-punc"`, with `return_raw_text=True` and `sentence_timestamp=True`.

Models are loaded lazily and cached via the existing single-active-engine manager; switching model/device closes the old adapter.

## Timestamp policy

FunASR's `timestamp` entries are millisecond pairs. They are not necessarily
one-to-one with the visible punctuated characters. The initial normalizer only
attributes times when raw token identity and count exactly match timestamps and
the normalized visible tokens. If not, word and character entries are preserved
without invented times, and warnings are returned. Multi-character Chinese word
time is aggregated from verified character spans. English/alphanumeric tokens
within Chinese speech can carry token-level time, without inventing times for
their component letters.

Jieba, if available, groups Chinese characters into words; otherwise each Han
character remains a conservative word for highlighting. Neither segmentation
path changes the underlying transcript text.

The adapter currently returns in-memory `partial` results, not exported files.
No real FunASR model was downloaded or tested on a GPU by CI.

Sources:
- https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/README_zh.md
- https://www.funasr.com/en/docs/quickstart.html
