"""Shared fixtures for the whispermlx test suite."""

import sys

# mlx requires Apple Silicon native libraries; stub it out on Linux/x86 CI
# so that test_asr.py (which imports whispermlx.asr → mlx_whisper) can be collected.
try:
    import mlx.core  # noqa: F401
except (ImportError, OSError):
    from unittest.mock import MagicMock

    for _mod in [
        "mlx",
        "mlx.core",
        "mlx_whisper",
        "mlx_whisper.audio",
        "mlx_whisper.decoding",
        "mlx_whisper.load_models",
    ]:
        sys.modules.setdefault(_mod, MagicMock())

import numpy as np
import pandas as pd
import pytest
import torch

SAMPLE_RATE = 16000


# ── Minimal segment class compatible with Vad.merge_chunks ────────────────────


class _Seg:
    """Minimal stand-in for diarize.Segment used by Vad.merge_chunks."""

    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker


# ── Audio fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def silence_audio():
    """1 second of silence as float32 numpy array at 16kHz."""
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def tone_audio():
    """1-second 440 Hz sine wave at 16kHz."""
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


@pytest.fixture
def multi_second_audio():
    """30 seconds of mixed signal — suitable for chunking tests."""
    t = np.linspace(0, 30, 30 * SAMPLE_RATE, endpoint=False)
    return (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)


@pytest.fixture
def vad_segments_factory():
    """Return a callable that builds lists of _Seg objects."""

    def factory(times):
        return [_Seg(s, e) for s, e in times]

    return factory


# ── Diarization fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def simple_diarize_df():
    """DataFrame with two non-overlapping speaker segments."""
    return pd.DataFrame(
        {
            "start": [0.0, 5.0],
            "end": [5.0, 10.0],
            "speaker": ["SPEAKER_00", "SPEAKER_01"],
        }
    )


@pytest.fixture
def overlapping_diarize_df():
    """DataFrame where speakers overlap slightly — tests intersection logic."""
    return pd.DataFrame(
        {
            "start": [0.0, 4.0],
            "end": [6.0, 10.0],
            "speaker": ["SPEAKER_00", "SPEAKER_01"],
        }
    )


# ── Transcript fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def minimal_transcript():
    """TranscriptionResult with two segments, no word-level data."""
    return {
        "segments": [
            {"start": 1.0, "end": 4.0, "text": "Hello world"},
            {"start": 6.0, "end": 9.0, "text": "How are you"},
        ],
        "language": "en",
    }


@pytest.fixture
def aligned_transcript():
    """AlignedTranscriptionResult with word-level timestamps."""
    return {
        "segments": [
            {
                "start": 1.0,
                "end": 4.0,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 1.0, "end": 2.5, "score": 0.9},
                    {"word": "world", "start": 2.5, "end": 4.0, "score": 0.85},
                ],
            },
            {
                "start": 6.0,
                "end": 9.0,
                "text": "How are you",
                "words": [
                    {"word": "How", "start": 6.0, "end": 7.0, "score": 0.8},
                    {"word": "are", "start": 7.0, "end": 8.0, "score": 0.75},
                    {"word": "you", "start": 8.0, "end": 9.0, "score": 0.9},
                ],
            },
        ],
        "word_segments": [
            {"word": "Hello", "start": 1.0, "end": 2.5, "score": 0.9},
            {"word": "world", "start": 2.5, "end": 4.0, "score": 0.85},
            {"word": "How", "start": 6.0, "end": 7.0, "score": 0.8},
            {"word": "are", "start": 7.0, "end": 8.0, "score": 0.75},
            {"word": "you", "start": 8.0, "end": 9.0, "score": 0.9},
        ],
    }


# ── Alignment fixtures ─────────────────────────────────────────────────────────

SMALL_DICT = {
    "<pad>": 0,
    "h": 1,
    "e": 2,
    "l": 3,
    "o": 4,
    "w": 5,
    "r": 6,
    "d": 7,
    "|": 8,  # word boundary
}


@pytest.fixture
def mock_align_model():
    """torchaudio-style mock that returns a fixed uniform emission."""
    from unittest.mock import MagicMock

    model = MagicMock()

    def _infer(waveform, lengths=None):
        frames = max(waveform.shape[-1] // 320, 2)
        vocab = max(SMALL_DICT.values()) + 1
        emission = torch.zeros(1, frames, vocab)
        return emission, None

    model.side_effect = _infer
    return model


@pytest.fixture
def align_metadata():
    return {"language": "en", "dictionary": SMALL_DICT, "type": "torchaudio"}


# ── MLX mock ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_mlx_result():
    """Return dict matching what _mlx_whisper_module.transcribe produces."""
    return {
        "text": "hello world",
        "language": "en",
        "segments": [
            {"text": "hello world", "tokens": [1, 2, 3], "avg_logprob": -0.3},
        ],
    }


# ── VAD mock ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vad_segments(vad_segments_factory):
    """Three VAD speech segments: 0-3s, 5-8s, 12-15s."""
    return vad_segments_factory([(0.0, 3.0), (5.0, 8.0), (12.0, 15.0)])
