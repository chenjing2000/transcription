"""Registry metadata for WhisperX; actual model binding is Phase 4."""
from transcription.result import EngineInfo

ENGINE_SPECS = (
    EngineInfo(
        name="whisperx_large_v3",
        model_id="large-v3",
        language="en",
        description="WhisperX large-v3 English with forced alignment (pending)",
    ),
)
