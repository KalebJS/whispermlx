import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch

from whispermlx.asr import MLXWhisperPipeline
from whispermlx.schema import AlignedTranscriptionResult
from whispermlx.schema import ProgressCallback
from whispermlx.schema import SingleSegment
from whispermlx.schema import TranscriptionResult
from whispermlx.vads import Vad

def load_align_model(
    language_code: str,
    device: str,
    model_name: str | None = ...,
    model_dir: str | None = ...,
    model_cache_only: bool = ...,
) -> tuple[torch.nn.Module, dict]: ...
def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: str | np.ndarray | torch.Tensor,
    device: str,
    interpolate_method: str = ...,
    return_char_alignments: bool = ...,
    print_progress: bool = ...,
    combined_progress: bool = ...,
    progress_callback: ProgressCallback = ...,
) -> AlignedTranscriptionResult: ...
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
def load_audio(file: str, sr: int = ...) -> np.ndarray: ...
def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: AlignedTranscriptionResult | TranscriptionResult,
    speaker_embeddings: dict[str, list[float]] | None = ...,
    fill_nearest: bool = ...,
) -> AlignedTranscriptionResult | TranscriptionResult: ...
def setup_logging(
    level: str = ...,
    log_file: str | None = ...,
) -> None: ...
def get_logger(name: str) -> logging.Logger: ...
