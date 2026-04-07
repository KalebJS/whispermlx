"""Tests for whispermlx/vads/ — Vad base class, Silero, and Pyannote helpers."""

import pytest
import torch

from whispermlx.vads.vad import Vad

# ── Minimal segment object for Vad.merge_chunks ───────────────────────────────


class _Seg:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker


# ── Vad base class ────────────────────────────────────────────────────────────


class TestVadBase:
    def test_valid_onset(self):
        vad = Vad(0.5)  # should not raise
        assert vad is not None

    def test_onset_zero_raises(self):
        with pytest.raises(ValueError, match="decimal value between 0 and 1"):
            Vad(0)

    def test_onset_one_raises(self):
        with pytest.raises(ValueError):
            Vad(1)

    def test_onset_above_one_raises(self):
        with pytest.raises(ValueError):
            Vad(1.1)

    def test_onset_negative_raises(self):
        with pytest.raises(ValueError):
            Vad(-0.1)

    def test_onset_near_zero_valid(self):
        Vad(0.001)  # just above 0, valid

    def test_onset_near_one_valid(self):
        Vad(0.999)  # just below 1, valid


# ── Vad.merge_chunks ──────────────────────────────────────────────────────────


class TestMergeChunks:
    def test_single_segment_under_chunk_size(self):
        segs = [_Seg(0.0, 5.0)]
        result = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.3)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 5.0

    def test_output_has_required_keys(self):
        segs = [_Seg(0.0, 5.0)]
        result = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.3)
        assert "start" in result[0]
        assert "end" in result[0]
        assert "segments" in result[0]

    def test_segments_list_contains_tuples(self):
        segs = [_Seg(0.0, 3.0), _Seg(5.0, 8.0)]
        result = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.3)
        assert isinstance(result[0]["segments"], list)
        assert isinstance(result[0]["segments"][0], tuple)

    def test_multiple_short_segments_merged(self):
        # 6 segments of 2s each = 12s total, chunk_size=30 → 1 chunk
        segs = [_Seg(float(i * 2), float(i * 2 + 2)) for i in range(6)]
        result = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.3)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 12.0

    def test_splits_when_exceeds_chunk_size(self):
        # 8 segments of 5s = 40s; chunk_size=30 → should produce at least 2 chunks
        segs = [_Seg(float(i * 5), float(i * 5 + 5)) for i in range(8)]
        result = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.3)
        assert len(result) >= 2

    def test_final_segment_always_included(self):
        # Even when total < chunk_size, the final accumulated group is appended
        segs = [_Seg(0.0, 1.0), _Seg(2.0, 3.0)]
        result = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.3)
        assert result[-1]["end"] == 3.0

    def test_chunk_boundaries_correct(self):
        segs = [_Seg(0.0, 20.0), _Seg(20.0, 40.0)]
        result = Vad.merge_chunks(segs, chunk_size=15, onset=0.5, offset=0.3)
        # first segment spans 0–20, which is > 15, so after one segment a new chunk starts
        # second segment spans 20–40, similarly > 15
        assert len(result) >= 2

    def test_speaker_attribute_used(self):
        segs = [_Seg(0.0, 5.0, "SPK_0")]
        result = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.3)
        assert result[0]["start"] == 0.0


# ── Silero ────────────────────────────────────────────────────────────────────


class TestSileroCall:
    """Test Silero.__call__ without triggering torch.hub.load."""

    @pytest.fixture
    def silero(self, monkeypatch):
        """Build a Silero instance with torch.hub.load mocked out."""
        fake_timestamps = []

        def fake_get_speech_timestamps(waveform, model, sampling_rate, **kwargs):
            return fake_timestamps

        fake_model = torch.nn.Module()
        fake_model.to = lambda device: fake_model

        monkeypatch.setattr(
            "torch.hub.load",
            lambda *a, **kw: (fake_model, (fake_get_speech_timestamps, None, None, None, None)),
        )

        from whispermlx.vads.silero import Silero

        instance = Silero(device="cpu", vad_onset=0.5, chunk_size=30, vad_offset=0.3)
        # Replace get_speech_timestamps so each test can customise the return
        instance._fake_timestamps = fake_timestamps
        instance.get_speech_timestamps = fake_get_speech_timestamps
        return instance, fake_timestamps

    def test_wrong_sample_rate_raises(self, silero):
        instance, _ = silero
        audio = {"waveform": torch.zeros(1, 16000), "sample_rate": 8000}
        with pytest.raises(ValueError, match="16000"):
            instance(audio)

    def test_correct_sample_rate_does_not_raise(self, silero, monkeypatch):
        instance, fake_ts = silero
        # Return no timestamps
        instance.get_speech_timestamps = lambda *a, **kw: []
        audio = {"waveform": torch.zeros(1, 16000), "sample_rate": 16000}
        result = instance(audio)
        assert result == []

    def test_returns_segment_objects_from_timestamps(self, silero):
        instance, _ = silero
        # 1s–2s in samples at 16kHz
        instance.get_speech_timestamps = lambda *a, **kw: [{"start": 16000, "end": 32000}]
        audio = {"waveform": torch.zeros(1, 32000), "sample_rate": 16000}
        result = instance(audio)
        assert len(result) == 1
        assert result[0].start == pytest.approx(1.0)
        assert result[0].end == pytest.approx(2.0)

    def test_empty_timestamps_returns_empty_list(self, silero):
        instance, _ = silero
        instance.get_speech_timestamps = lambda *a, **kw: []
        audio = {"waveform": torch.zeros(1, 16000), "sample_rate": 16000}
        result = instance(audio)
        assert result == []

    def test_multiple_timestamps_converted(self, silero):
        instance, _ = silero
        instance.get_speech_timestamps = lambda *a, **kw: [
            {"start": 0, "end": 16000},
            {"start": 32000, "end": 48000},
        ]
        audio = {"waveform": torch.zeros(1, 48000), "sample_rate": 16000}
        result = instance(audio)
        assert len(result) == 2
        assert result[0].start == pytest.approx(0.0)
        assert result[1].start == pytest.approx(2.0)


class TestSileroCallNumpyWaveform:
    """Regression tests for the NumPy waveform bug fixed in PR #3."""

    @pytest.fixture
    def silero(self, monkeypatch):
        fake_model = torch.nn.Module()
        fake_model.to = lambda device: fake_model

        monkeypatch.setattr(
            "torch.hub.load",
            lambda *a, **kw: (fake_model, (lambda *a, **kw: [], None, None, None, None)),
        )
        from whispermlx.vads.silero import Silero

        instance = Silero(device="cpu", vad_onset=0.5, chunk_size=30, vad_offset=0.3)
        instance.get_speech_timestamps = lambda *a, **kw: []
        return instance

    def test_numpy_waveform_does_not_raise(self, silero):
        import numpy as np

        audio = {"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000}
        result = silero(audio)
        assert result == []

    def test_numpy_waveform_returns_segments(self, silero):
        import numpy as np

        silero.get_speech_timestamps = lambda *a, **kw: [{"start": 0, "end": 16000}]
        audio = {"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000}
        result = silero(audio)
        assert len(result) == 1
        assert result[0].start == pytest.approx(0.0)
        assert result[0].end == pytest.approx(1.0)

    def test_numpy_float64_waveform_converted_to_float32(self, silero, monkeypatch):
        import numpy as np

        captured = {}

        def capturing_get_speech_timestamps(waveform, **kwargs):
            captured["waveform"] = waveform
            return []

        silero.get_speech_timestamps = capturing_get_speech_timestamps
        audio = {"waveform": np.zeros(16000, dtype=np.float64), "sample_rate": 16000}
        silero(audio)
        assert torch.is_tensor(captured["waveform"])
        assert captured["waveform"].dtype == torch.float32


class TestSileroPreprocessAudio:
    @pytest.fixture
    def silero_class(self, monkeypatch):
        fake_model = torch.nn.Module()
        fake_model.to = lambda device: fake_model
        monkeypatch.setattr(
            "torch.hub.load",
            lambda *a, **kw: (fake_model, (lambda *a, **kw: [], None, None, None, None)),
        )
        from whispermlx.vads.silero import Silero

        return Silero

    def test_numpy_array_returns_float32_tensor(self, silero_class):
        import numpy as np

        audio = np.zeros(16000, dtype=np.float32)
        result = silero_class.preprocess_audio(audio)
        assert torch.is_tensor(result)
        assert result.dtype == torch.float32

    def test_numpy_float64_cast_to_float32(self, silero_class):
        import numpy as np

        audio = np.zeros(16000, dtype=np.float64)
        result = silero_class.preprocess_audio(audio)
        assert torch.is_tensor(result)
        assert result.dtype == torch.float32

    def test_tensor_passthrough_stays_float32(self, silero_class):
        audio = torch.zeros(16000, dtype=torch.float32)
        result = silero_class.preprocess_audio(audio)
        assert torch.is_tensor(result)
        assert result.dtype == torch.float32

    def test_tensor_non_float32_cast_to_float32(self, silero_class):
        audio = torch.zeros(16000, dtype=torch.float64)
        result = silero_class.preprocess_audio(audio)
        assert result.dtype == torch.float32


class TestSileroMergeChunks:
    @pytest.fixture
    def silero_class(self, monkeypatch):
        fake_model = torch.nn.Module()
        fake_model.to = lambda device: fake_model
        monkeypatch.setattr(
            "torch.hub.load",
            lambda *a, **kw: (fake_model, (lambda *a, **kw: [], None, None, None, None)),
        )
        from whispermlx.vads.silero import Silero

        return Silero

    def test_empty_list_returns_empty(self, silero_class):
        result = silero_class.merge_chunks([], chunk_size=30, onset=0.5, offset=0.3)
        assert result == []

    def test_zero_chunk_size_raises(self, silero_class):
        segs = [_Seg(0.0, 1.0)]
        with pytest.raises(AssertionError):
            silero_class.merge_chunks(segs, chunk_size=0, onset=0.5, offset=0.3)

    def test_non_empty_delegates_to_vad(self, silero_class):
        segs = [_Seg(0.0, 5.0)]
        result = silero_class.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.3)
        assert len(result) == 1
        assert result[0]["start"] == 0.0


# ── Pyannote.preprocess_audio ─────────────────────────────────────────────────


class TestPyannotePreprocessAudio:
    def test_numpy_to_2d_tensor(self):
        import numpy as np

        from whispermlx.vads.pyannote import Pyannote

        audio = np.ones(16000, dtype=np.float32)
        result = Pyannote.preprocess_audio(audio)
        assert torch.is_tensor(result)
        assert result.ndim == 2
        assert result.shape == (1, 16000)

    def test_output_preserves_dtype(self):
        import numpy as np

        from whispermlx.vads.pyannote import Pyannote

        # preprocess_audio uses torch.from_numpy which preserves dtype
        audio = np.ones(1000, dtype=np.float32)
        result = Pyannote.preprocess_audio(audio)
        assert result.dtype == torch.float32

    def test_float64_input_preserved(self):
        import numpy as np

        from whispermlx.vads.pyannote import Pyannote

        audio = np.ones(1000, dtype=np.float64)
        result = Pyannote.preprocess_audio(audio)
        assert result.dtype == torch.float64
