from collections.abc import Callable

import numpy as np
import torch
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation
from pyannote.core import SlidingWindowFeature

from whispermlx.vads.vad import Vad

def load_vad_model(
    device: str | torch.device,
    vad_onset: float = ...,
    vad_offset: float = ...,
    token: str | None = ...,
    model_fp: str | None = ...,
) -> VoiceActivitySegmentation: ...

class Binarize:
    onset: float
    offset: float
    pad_onset: float
    pad_offset: float
    min_duration_on: float
    min_duration_off: float
    max_duration: float

    def __init__(
        self,
        onset: float = ...,
        offset: float | None = ...,
        min_duration_on: float = ...,
        min_duration_off: float = ...,
        pad_onset: float = ...,
        pad_offset: float = ...,
        max_duration: float = ...,
    ) -> None: ...
    def __call__(self, scores: SlidingWindowFeature) -> Annotation: ...

class VoiceActivitySegmentation(VoiceActivityDetection):
    def __init__(
        self,
        segmentation: PipelineModel = ...,
        fscore: bool = ...,
        token: str | None = ...,
        **inference_kwargs: object,
    ) -> None: ...
    def apply(
        self,
        file: AudioFile,
        hook: Callable | None = ...,
    ) -> Annotation: ...

class Pyannote(Vad):
    vad_pipeline: VoiceActivitySegmentation

    def __init__(
        self,
        device: str | torch.device,
        token: str | None = ...,
        model_fp: str | None = ...,
        **kwargs: object,
    ) -> None: ...
    def __call__(self, audio: AudioFile, **kwargs: object) -> list: ...
    @staticmethod
    def preprocess_audio(audio: np.ndarray) -> torch.Tensor: ...
    @staticmethod
    def merge_chunks(
        segments: object,
        chunk_size: int,
        onset: float = ...,
        offset: float | None = ...,
    ) -> list[dict]: ...
