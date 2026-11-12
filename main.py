"""Example usage of the public Python API (does not perform ASR yet)."""
from transcription import list_engines, transcribe, release_models


def main() -> None:
    print("Registered engines:")
    for item in list_engines():
        print(f"  {item.name} ({item.language}, model={item.model_id})")
    # Real transcription is deliberately not implemented in this foundation.
    # Example once Phase 4/5 are complete:
    # result = transcribe(r"D:\Videos\lecture.mp4", engine="whisperx_large_v3")
    # print(result.status, result.info, result.files)
    release_models()


if __name__ == "__main__":
    main()
