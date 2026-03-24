import numpy as np

from whispermlx.schema import ProgressCallback
from whispermlx.schema import TranscriptionResult
from whispermlx.vads import Vad

MLX_MODEL_MAP: dict[str, str]

class MLXWhisperPipeline:
    model_path: str
    vad_model: object
    preset_language: str | None
    task: str
    initial_prompt: str | None

    def __init__(
        self,
        model_path: str,
        vad: object,
        vad_params: dict,
        language: str | None = ...,
        task: str = ...,
        initial_prompt: str | None = ...,
    ) -> None: ...
    def transcribe(
        self,
        audio: str | np.ndarray,
        batch_size: int | None = ...,
        num_workers: int = ...,
        language: str | None = ...,
        task: str | None = ...,
        chunk_size: int = ...,
        print_progress: bool = ...,
        combined_progress: bool = ...,
        verbose: bool = ...,
        progress_callback: ProgressCallback = ...,
    ) -> TranscriptionResult: ...

def load_model(
    whisper_arch: str,
    device: str,
    device_index: int = ...,
    compute_type: str = ...,
    asr_options: dict | None = ...,
    language: str | None = ...,
    vad_model: Vad | None = ...,
    vad_method: str | None = ...,
    vad_options: dict | None = ...,
    model: object = ...,
    task: str = ...,
    download_root: str | None = ...,
    local_files_only: bool = ...,
    threads: int = ...,
    use_auth_token: object = ...,
) -> MLXWhisperPipeline: ...
