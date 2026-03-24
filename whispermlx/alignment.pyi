from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from whispermlx.schema import AlignedTranscriptionResult
from whispermlx.schema import ProgressCallback
from whispermlx.schema import SingleSegment

DEFAULT_ALIGN_MODELS_TORCH: dict[str, str]
DEFAULT_ALIGN_MODELS_HF: dict[str, str]
LANGUAGES_WITHOUT_SPACES: list[str]

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
def get_trellis(
    emission: torch.Tensor,
    tokens: list[int],
    blank_id: int = ...,
) -> torch.Tensor: ...
@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(
    trellis: torch.Tensor,
    emission: torch.Tensor,
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
