# Test and acceptance boundaries

## Routine Windows CI

`.github/workflows/python-tests.yml` tests Python 3.11 and 3.12.
Tests cover metadata, adapter mocks, model lifecycle, FFmpeg stream-copy,
canonical text spans, grouped JSON/TXT output, failed writes and publishing
contention. Four opt-in tests are skipped by default; CI never silently
downloads multi-gigabyte ASR models.

## Previously successful real CPU tests

The workflows `whisperx-cpu-smoke.yml` and `real-speech-cpu.yml`
have passed on Windows with short real English and Mandarin audio.
The English speech test uses WhisperX `tiny.en`; the Mandarin sample uses
FunASR Paraformer. They test functioning recognition and output, not
word-error-rate benchmarks or English large-v3 GPU readiness.

The optional dependency smoke workflow has separately verified that
WhisperX and FunASR can be installed and imported together on a Windows
Python 3.11 runner.

## Local GPU test (not yet executed)

Use `python -m transcription.doctor` and
`python -m transcription.gpu_diagnostics` to inspect the environment,
then `scripts/run-gpu-acceptance.ps1` for opt-in bilingual
large-v3 -> Paraformer -> large-v3 switching. See
`docs/hardware_acceptance.md` for steps and generated reports.
No successful CPU runner run proves CUDA support or VRAM release.

## Known limitations

- Word and character timestamps remain missing when the model alignment
  cannot verify them; recorded candidate video offsets are not guaranteed
  playback synchronization for edit lists or discontinuities.
- The publisher stages files, uses a cooperative directory lock and
  exclusive creation, and rolls back ordinary write failures.
  A sudden process crash can still leave a partial file group or stale lock.
- Sentence splitting uses lightweight punctuation rules, with limited
  exceptions for decimals, titles and English initials; it is not a full
  linguistic sentence boundary detector.
