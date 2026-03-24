from collections.abc import Mapping
from io import IOBase
from pathlib import Path

import torch

from whispermlx.diarize import Segment
from whispermlx.vads.vad import Vad

AudioFile = str | Path | IOBase | Mapping

class Silero(Vad):
    vad_onset: float
    chunk_size: int
    device: torch.device

    def __init__(self, **kwargs: object) -> None: ...
    def __call__(self, audio: AudioFile, **kwargs: object) -> list[Segment]: ...
    @staticmethod
    def preprocess_audio(audio: object) -> object: ...
    @staticmethod
    def merge_chunks(
        segments_list: list,
        chunk_size: int,
        onset: float = ...,
        offset: float | None = ...,
    ) -> list[dict]: ...
