"""Minimal public API example.

The default invocation only lists declared engines; it never downloads models.
To run recognition, install a compatible ASR framework and uncomment an example.
"""
from transcription import list_engines, transcribe, release_models


def main() -> None:
    for item in list_engines():
        print(f"{item.name} | {item.language} | {item.model_id}")
    # After installing WhisperX/FunASR and FFmpeg:
    # result = transcribe(r"D:\Videos\lecture.mp4", engine="whisperx_large_v3")
    # print(result.status, result.info, result.files)
    release_models()


if __name__ == "__main__":
    main()
