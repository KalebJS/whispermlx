import numpy as np
import torch

SAMPLE_RATE: int
N_FFT: int
HOP_LENGTH: int
CHUNK_LENGTH: int
N_SAMPLES: int
N_FRAMES: int
N_SAMPLES_PER_TOKEN: int
FRAMES_PER_SECOND: int
TOKENS_PER_SECOND: int

def load_audio(file: str, sr: int = ...) -> np.ndarray: ...
def pad_or_trim(
    array: np.ndarray | torch.Tensor,
    length: int = ...,
    *,
    axis: int = ...,
) -> np.ndarray | torch.Tensor: ...
def mel_filters(device: torch.device | str, n_mels: int) -> torch.Tensor: ...
def log_mel_spectrogram(
    audio: str | np.ndarray | torch.Tensor,
    n_mels: int,
    padding: int = ...,
    device: str | torch.device | None = ...,
) -> torch.Tensor: ...
