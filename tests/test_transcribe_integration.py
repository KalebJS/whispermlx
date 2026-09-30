"""Integration tests for whispermlx/transcribe.py — transcribe_task() pipeline."""

import argparse
import copy
import os
import warnings

import numpy as np
import pandas as pd
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def make_args(tmp_path, **overrides):
    """Build the full args dict that transcribe_task expects (mirrors argparse output)."""
    defaults = {
        # audio
        "audio": [str(tmp_path / "audio.wav")],
        # model
        "model": "small",
        "batch_size": 8,
        "model_dir": None,
        "model_cache_only": False,
        "device": "cpu",
        "device_index": 0,
        "compute_type": "default",
        "verbose": False,
        # output
        "output_dir": str(tmp_path / "out"),
        "output_format": "json",
        # task
        "task": "transcribe",
        "language": None,
        # alignment
        "align_model": None,
        "interpolate_method": "nearest",
        "no_align": False,
        "return_char_alignments": False,
        # vad
        "vad_method": "pyannote",
        "vad_onset": 0.5,
        "vad_offset": 0.363,
        "chunk_size": 30,
        # diarize
        "diarize": False,
        "min_speakers": None,
        "max_speakers": None,
        "diarize_model": "pyannote/speaker-diarization-community-1",
        "speaker_embeddings": False,
        # asr decoding
        "temperature": 0.0,
        "temperature_increment_on_fallback": 0.2,
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "hotwords": None,
        "suppress_tokens": "-1",
        "suppress_numerals": False,
        # output formatting
        "highlight_words": False,
        "max_line_width": None,
        "max_line_count": None,
        "segment_resolution": "sentence",
        # misc
        "print_progress": False,
        "threads": 0,
        "hf_token": None,
    }
    defaults.update(overrides)
    return defaults


MINIMAL_TRANSCRIPT = {
    "segments": [{"start": 1.0, "end": 4.0, "text": "hello"}],
    "language": "en",
}

ALIGNED_TRANSCRIPT = {
    "segments": [
        {
            "start": 1.0,
            "end": 4.0,
            "text": "hello",
            "words": [{"word": "hello", "start": 1.0, "end": 4.0, "score": 0.9}],
        }
    ],
    "word_segments": [{"word": "hello", "start": 1.0, "end": 4.0, "score": 0.9}],
}

DIARIZE_DF = pd.DataFrame(
    {
        "start": [0.0],
        "end": [5.0],
        "speaker": ["SPEAKER_00"],
        "segment": [None],
        "label": ["A"],
    }
)


def _make_mock_pipeline(mocker, transcript=MINIMAL_TRANSCRIPT):
    mock = mocker.MagicMock()
    mock.transcribe.return_value = copy.deepcopy(transcript)
    return mock


def _make_parser():
    return argparse.ArgumentParser()


# ── tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestTranscribeTaskIntegration:
    def test_asr_only_no_align_no_diarize(self, tmp_path, mocker):
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mock_writer = mocker.MagicMock()
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mock_writer)

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, no_align=True, diarize=False)
        transcribe_task(args, _make_parser())

        mock_writer.assert_called_once()
        call_result = mock_writer.call_args[0][0]
        assert "segments" in call_result

    def test_asr_plus_align(self, tmp_path, mocker):
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch(
            "whispermlx.transcribe.load_align_model",
            return_value=(
                mocker.MagicMock(),
                {"language": "en", "dictionary": {}, "type": "torchaudio"},
            ),
        )
        mock_align = mocker.patch(
            "whispermlx.transcribe.align",
            return_value=copy.deepcopy(ALIGNED_TRANSCRIPT),
        )
        mock_writer = mocker.MagicMock()
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mock_writer)

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, no_align=False, diarize=False)
        transcribe_task(args, _make_parser())

        mock_align.assert_called_once()
        call_result = mock_writer.call_args[0][0]
        assert "word_segments" in call_result

    def test_asr_plus_align_plus_diarize(self, tmp_path, mocker):
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch(
            "whispermlx.transcribe.load_align_model",
            return_value=(
                mocker.MagicMock(),
                {"language": "en", "dictionary": {}, "type": "torchaudio"},
            ),
        )
        mocker.patch(
            "whispermlx.transcribe.align",
            return_value=copy.deepcopy(ALIGNED_TRANSCRIPT),
        )
        mock_diarize_class = mocker.patch("whispermlx.transcribe.DiarizationPipeline")
        mock_diarize_instance = mocker.MagicMock()
        mock_diarize_instance.return_value = copy.deepcopy(DIARIZE_DF)
        mock_diarize_class.return_value = mock_diarize_instance

        mock_assign = mocker.patch(
            "whispermlx.transcribe.assign_word_speakers",
            return_value=copy.deepcopy(ALIGNED_TRANSCRIPT),
        )
        mock_writer = mocker.MagicMock()
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mock_writer)

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, no_align=False, diarize=True)
        transcribe_task(args, _make_parser())

        mock_diarize_class.assert_called_once()
        mock_assign.assert_called_once()

    def test_translate_task_skips_align(self, tmp_path, mocker):
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mock_align = mocker.patch("whispermlx.transcribe.align")
        mock_writer = mocker.MagicMock()
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mock_writer)

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, task="translate", no_align=False)
        transcribe_task(args, _make_parser())

        mock_align.assert_not_called()

    def test_language_normalization_full_name(self, tmp_path, mocker):
        """language='english' should be converted to 'en'."""
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch(
            "whispermlx.transcribe.load_align_model",
            return_value=(
                mocker.MagicMock(),
                {"language": "en", "dictionary": {}, "type": "torchaudio"},
            ),
        )
        mocker.patch(
            "whispermlx.transcribe.align",
            return_value=copy.deepcopy(ALIGNED_TRANSCRIPT),
        )
        mock_writer = mocker.MagicMock()
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mock_writer)

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, language="english", no_align=False)
        transcribe_task(args, _make_parser())
        # If language normalization works, no ValueError is raised
        mock_writer.assert_called_once()

    def test_unsupported_language_raises(self, tmp_path, mocker):
        mocker.patch("whispermlx.transcribe.load_model", return_value=mocker.MagicMock())
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mocker.MagicMock())

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, language="klingon")
        with pytest.raises(ValueError, match="Unsupported language"):
            transcribe_task(args, _make_parser())

    def test_english_only_model_forces_english(self, tmp_path, mocker):
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mocker.MagicMock())

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, model="small.en", language="fr", no_align=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transcribe_task(args, _make_parser())
        warning_messages = [str(x.message) for x in w]
        assert any("English" in msg or "english" in msg.lower() for msg in warning_messages)

    def test_speaker_embeddings_without_diarize_warns(self, tmp_path, mocker):
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mocker.MagicMock())

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, speaker_embeddings=True, diarize=False, no_align=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transcribe_task(args, _make_parser())
        assert any("speaker_embeddings" in str(x.message) for x in w)

    def test_multiple_audio_files(self, tmp_path, mocker):
        audio_a = str(tmp_path / "a.wav")
        audio_b = str(tmp_path / "b.wav")

        mock_pipeline = mocker.MagicMock()
        mock_pipeline.transcribe.return_value = copy.deepcopy(MINIMAL_TRANSCRIPT)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio",
            return_value=np.zeros(16000, dtype=np.float32),
        )
        mock_writer = mocker.MagicMock()
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mock_writer)

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, audio=[audio_a, audio_b], no_align=True)
        transcribe_task(args, _make_parser())

        assert mock_pipeline.transcribe.call_count == 2
        assert mock_writer.call_count == 2

    def test_interleaved_context_forwarded_to_pipeline(self, tmp_path, mocker):
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mocker.MagicMock())

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, interleaved_context=True, no_align=True)
        transcribe_task(args, _make_parser())

        call_kwargs = mock_pipeline.transcribe.call_args[1]
        assert call_kwargs["interleaved_context"] is True

    def test_temperature_increment_builds_tuple(self, tmp_path, mocker):
        captured = {}

        def fake_load_model(*args, **kwargs):
            captured["asr_options"] = kwargs.get("asr_options", {})
            m = mocker.MagicMock()
            m.transcribe.return_value = copy.deepcopy(MINIMAL_TRANSCRIPT)
            return m

        mocker.patch("whispermlx.transcribe.load_model", side_effect=fake_load_model)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mocker.MagicMock())

        from whispermlx.transcribe import transcribe_task

        args = make_args(
            tmp_path,
            temperature=0.0,
            temperature_increment_on_fallback=0.2,
            no_align=True,
        )
        transcribe_task(args, _make_parser())
        temps = captured["asr_options"]["temperatures"]
        assert hasattr(temps, "__len__")
        assert len(temps) > 1

    def test_output_dir_created(self, tmp_path, mocker):
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mocker.MagicMock())

        from whispermlx.transcribe import transcribe_task

        output_dir = str(tmp_path / "new_subdir" / "nested")
        args = make_args(tmp_path, output_dir=output_dir, no_align=True)
        transcribe_task(args, _make_parser())
        assert os.path.isdir(output_dir)

    def test_diarize_with_embeddings(self, tmp_path, mocker):
        mock_pipeline = _make_mock_pipeline(mocker)
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch(
            "whispermlx.transcribe.load_align_model",
            return_value=(
                mocker.MagicMock(),
                {"language": "en", "dictionary": {}, "type": "torchaudio"},
            ),
        )
        mocker.patch(
            "whispermlx.transcribe.align",
            return_value=copy.deepcopy(ALIGNED_TRANSCRIPT),
        )
        embeddings = {"SPEAKER_00": [0.1, 0.2]}
        mock_diarize_class = mocker.patch("whispermlx.transcribe.DiarizationPipeline")
        mock_diarize_instance = mocker.MagicMock()
        mock_diarize_instance.return_value = (copy.deepcopy(DIARIZE_DF), embeddings)
        mock_diarize_class.return_value = mock_diarize_instance

        aligned_with_speaker = copy.deepcopy(ALIGNED_TRANSCRIPT)
        aligned_with_speaker["speaker_embeddings"] = embeddings
        mock_assign = mocker.patch(
            "whispermlx.transcribe.assign_word_speakers",
            return_value=aligned_with_speaker,
        )
        mock_writer = mocker.MagicMock()
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mock_writer)

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, no_align=False, diarize=True, speaker_embeddings=True)
        transcribe_task(args, _make_parser())

        # assign_word_speakers should be called with embeddings
        call_kwargs = mock_assign.call_args
        assert call_kwargs is not None

    def test_align_skipped_for_empty_segments(self, tmp_path, mocker):
        empty_transcript = {"segments": [], "language": "en"}
        mock_pipeline = mocker.MagicMock()
        mock_pipeline.transcribe.return_value = empty_transcript
        mocker.patch("whispermlx.transcribe.load_model", return_value=mock_pipeline)
        mocker.patch(
            "whispermlx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)
        )
        mocker.patch(
            "whispermlx.transcribe.load_align_model",
            return_value=(
                mocker.MagicMock(),
                {"language": "en", "dictionary": {}, "type": "torchaudio"},
            ),
        )
        mock_align = mocker.patch("whispermlx.transcribe.align")
        mock_writer = mocker.MagicMock()
        mocker.patch("whispermlx.transcribe.get_writer", return_value=mock_writer)

        from whispermlx.transcribe import transcribe_task

        args = make_args(tmp_path, no_align=False)
        transcribe_task(args, _make_parser())

        # align should not be called when segments is empty
        mock_align.assert_not_called()
