"""Registry metadata for FunASR; actual model binding is Phase 5."""
from transcription.result import EngineInfo

ENGINE_SPECS = (
    EngineInfo(
        name="paraformer_zh",
        model_id="paraformer-zh",
        language="zh",
        description="FunASR Paraformer Chinese, VAD and punctuation (pending)",
    ),
)
