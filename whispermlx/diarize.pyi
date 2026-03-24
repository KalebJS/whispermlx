import numpy as np
import pandas as pd
import torch

from whispermlx.schema import AlignedTranscriptionResult
from whispermlx.schema import ProgressCallback
from whispermlx.schema import TranscriptionResult

class IntervalTree:
    starts: np.ndarray
    ends: np.ndarray
    speakers: list[str]

    def __init__(self, intervals: list[tuple[float, float, str]]) -> None: ...
    def query(self, start: float, end: float) -> list[tuple[str, float]]: ...
    def find_nearest(self, time: float) -> str | None: ...

class DiarizationPipeline:
    def __init__(
        self,
        model_name: str | None = ...,
        token: str | None = ...,
        device: str | torch.device | None = ...,
        cache_dir: str | None = ...,
    ) -> None: ...
    def __call__(
        self,
        audio: str | np.ndarray,
        num_speakers: int | None = ...,
        min_speakers: int | None = ...,
        max_speakers: int | None = ...,
        return_embeddings: bool = ...,
        progress_callback: ProgressCallback = ...,
    ) -> tuple[pd.DataFrame, dict[str, list[float]] | None] | pd.DataFrame: ...

def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: AlignedTranscriptionResult | TranscriptionResult,
    speaker_embeddings: dict[str, list[float]] | None = ...,
    fill_nearest: bool = ...,
) -> AlignedTranscriptionResult | TranscriptionResult: ...

class Segment:
    start: int
    end: int
    speaker: str | None

    def __init__(self, start: int, end: int, speaker: str | None = ...) -> None: ...
