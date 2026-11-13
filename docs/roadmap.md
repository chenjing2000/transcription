# Implementation and validation status

| Area | Current status |
| --- | --- |
| Public API, engine registry, single active model manager | Implemented; Windows unit tests passed |
| FFprobe and FFmpeg stream-copy audio extraction | Implemented; Windows integration tests passed |
| English WhisperX adapter with forced word alignment | Implemented; short real `tiny.en` speech CPU test passed |
| Chinese FunASR Paraformer with VAD/punctuation | Implemented; short real Chinese speech CPU test passed |
| Four JSON/TXT outputs and video original-codec audio | Implemented; publishing tests passed |
| Error handling, locking, collision-safe grouped numbering | Implemented; ordinary failure and contention tests |
| RTX 5060 Ti CUDA execution, large-v3 model loading, bilingual GPU switching | Pending hardware acceptance |
| Video edit lists/discontinuous media-clock synchronization | Pending specialized media validation |
| Power-loss/crash-atomic multi-file publishing | Not implemented; documented limitation |

The short CPU speech tests do not constitute acoustic accuracy benchmarks,
long-recording stability tests, or GPU acceptance.

Code was simplified in three passes without changing the public API or
published JSON schema. The first two passes passed Windows CI; the third
adds sentence-edge and publish-contention regression tests.
