from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from typing import Protocol

import numpy as np

from whispermlx.schema import AlignedTranscriptionResult
from whispermlx.schema import ProgressCallback
from whispermlx.schema import SingleSegment

DEFAULT_ALIGN_MODELS_TORCH: dict[str, str]
DEFAULT_ALIGN_MODELS_HF: dict[str, str]
TORCHAUDIO_PIPELINE_TO_HF: dict[str, str]
LANGUAGES_WITHOUT_SPACES: list[str]

class AlignModel(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def load_align_model(
    language_code: str,
    device: str,
    model_name: str | None = ...,
    model_dir: str | None = ...,
    model_cache_only: bool = ...,
    backend: str = ...,
) -> tuple[AlignModel, dict]: ...
def align(
    transcript: Iterable[SingleSegment],
    model: AlignModel,
    align_model_metadata: dict,
    audio: str | np.ndarray,
    device: str,
    interpolate_method: str = ...,
    return_char_alignments: bool = ...,
    print_progress: bool = ...,
    combined_progress: bool = ...,
    progress_callback: ProgressCallback = ...,
) -> AlignedTranscriptionResult: ...
def get_trellis(
    emission: np.ndarray,
    tokens: list[int],
    blank_id: int = ...,
) -> np.ndarray: ...
@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(
    trellis: np.ndarray,
    emission: np.ndarray,
    tokens: list[int],
    blank_id: int = ...,
) -> list[Point] | None: ...
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    @property
    def length(self) -> int: ...

def merge_repeats(path: list[Point], transcript: str) -> list[Segment]: ...
def merge_words(segments: list[Segment], separator: str = ...) -> list[Segment]: ...
