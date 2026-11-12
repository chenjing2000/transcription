"""Expected task-level error with a stable machine-readable code."""


class TranscriptionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)
