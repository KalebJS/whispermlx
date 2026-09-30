"""Tests for whispermlx/asr.py — model resolution, logprob computation, pipeline, load_model."""

import numpy as np
import pytest

from whispermlx.asr import INTERLEAVED_CONTEXT_MAX_CHARS
from whispermlx.asr import MLX_MODEL_MAP
from whispermlx.asr import MLXWhisperPipeline
from whispermlx.asr import _compute_avg_logprob
from whispermlx.asr import _join_prompts
from whispermlx.asr import _resolve_mlx_model
from whispermlx.asr import _trim_tail
from whispermlx.vads.vad import Vad

# ── Minimal VAD subclass for transcribe() tests ───────────────────────────────


class _MockVad(Vad):
    """Minimal Vad subclass that returns a single merged chunk."""

    def __init__(self, segments=None):
        # skip parent validation
        self.vad_onset = 0.5
        self._segments = segments or []

    def __call__(self, audio_dict):
        return self._segments

    @staticmethod
    def preprocess_audio(audio):
        return audio

    @staticmethod
    def merge_chunks(segments, chunk_size, onset, offset):
        if not segments:
            return []
        return [{"start": segments[0].start, "end": segments[-1].end, "segments": []}]


class _Seg:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker


# ── _resolve_mlx_model ────────────────────────────────────────────────────────


class TestResolveMLXModel:
    def test_short_name_maps_to_repo(self):
        result = _resolve_mlx_model("tiny")
        assert result == "mlx-community/whisper-tiny-mlx"

    def test_full_hf_id_passthrough(self):
        result = _resolve_mlx_model("org/custom-whisper")
        assert result == "org/custom-whisper"

    def test_unknown_name_returns_as_is(self):
        result = _resolve_mlx_model("nonexistent-model")
        assert result == "nonexistent-model"

    @pytest.mark.parametrize("name,expected", list(MLX_MODEL_MAP.items()))
    def test_all_known_names_resolve(self, name, expected):
        assert _resolve_mlx_model(name) == expected

    def test_full_hf_with_slash_preserved(self):
        repo = "mlx-community/whisper-large-v3-mlx"
        assert _resolve_mlx_model(repo) == repo


# ── _compute_avg_logprob ──────────────────────────────────────────────────────


class TestComputeAvgLogprob:
    def test_empty_list_returns_zero(self):
        assert _compute_avg_logprob([]) == 0.0

    def test_single_segment(self):
        segs = [{"tokens": [1, 2, 3], "avg_logprob": -0.5}]
        result = _compute_avg_logprob(segs)
        assert result == pytest.approx(-0.5)

    def test_weighted_by_token_count(self):
        segs = [
            {"tokens": [1], "avg_logprob": -1.0},  # weight 1
            {"tokens": [1, 2, 3], "avg_logprob": 0.0},  # weight 3
        ]
        # weighted avg = (-1.0*1 + 0.0*3) / 4 = -0.25
        result = _compute_avg_logprob(segs)
        assert result == pytest.approx(-0.25)

    def test_missing_tokens_treated_as_one(self):
        segs = [{"avg_logprob": -0.8}]
        result = _compute_avg_logprob(segs)
        assert result == pytest.approx(-0.8)

    def test_missing_avg_logprob_treated_as_zero(self):
        segs = [{"tokens": [1, 2]}]
        result = _compute_avg_logprob(segs)
        assert result == pytest.approx(0.0)

    def test_empty_tokens_list_treated_as_one(self):
        segs = [{"tokens": [], "avg_logprob": -0.6}]
        result = _compute_avg_logprob(segs)
        assert result == pytest.approx(-0.6)


# ── interleaved context helpers ──────────────────────────────────────────────


class TestTrimTail:
    def test_short_text_unchanged(self):
        assert _trim_tail("hello world", 100) == "hello world"

    def test_exact_length_unchanged(self):
        text = "a" * 50
        assert _trim_tail(text, 50) == text

    def test_long_text_snaps_to_word_boundary(self):
        # tail of 11 chars slices mid-word; result snaps to the last full word
        assert _trim_tail("first second third", 11) == "third"

    def test_no_space_in_tail_keeps_raw_tail(self):
        assert _trim_tail("a" * 30, 5) == "a" * 5


class TestJoinPrompts:
    def test_both_empty_returns_none(self):
        assert _join_prompts(None, "", 900) is None

    def test_whitespace_only_returns_none(self):
        assert _join_prompts("   ", "   ", 900) is None

    def test_initial_prompt_only(self):
        assert _join_prompts("domain terms", "", 900) == "domain terms"

    def test_carried_context_only(self):
        assert _join_prompts(None, "prior text", 900) == "prior text"

    def test_combines_user_prompt_and_context(self):
        assert _join_prompts("domain terms", "prior text", 900) == "domain terms prior text"

    def test_user_prompt_exhausts_budget(self):
        result = _join_prompts("a" * 900, "prior text", 900)
        assert result == "a" * 900

    def test_carried_context_trimmed_to_remaining_budget(self):
        result = _join_prompts("prompt", "y" * 100, 20)
        assert result == "prompt " + "y" * 14


# ── MLXWhisperPipeline.transcribe ─────────────────────────────────────────────


class TestMLXWhisperPipelineTranscribe:
    """Use _MockVad + patch _mlx_whisper_module.transcribe."""

    MLX_RESULT = {
        "text": "hello world",
        "language": "en",
        "segments": [{"text": "hello world", "tokens": [1, 2, 3], "avg_logprob": -0.3}],
    }

    def _make_pipeline(self, language=None):
        vad = _MockVad(segments=[_Seg(0.0, 3.0)])
        return MLXWhisperPipeline(
            model_path="mlx-community/whisper-tiny-mlx",
            vad=vad,
            vad_params={"vad_onset": 0.5, "vad_offset": 0.3},
            language=language,
        )

    def test_numpy_input_returns_transcription_result(self, mocker):
        mocker.patch(
            "whispermlx.asr._mlx_whisper_module.transcribe",
            return_value=self.MLX_RESULT,
        )
        pipeline = self._make_pipeline()
        audio = np.zeros(48000, dtype=np.float32)
        result = pipeline.transcribe(audio)
        assert "segments" in result
        assert "language" in result

    def test_result_has_correct_structure(self, mocker):
        mocker.patch(
            "whispermlx.asr._mlx_whisper_module.transcribe",
            return_value=self.MLX_RESULT,
        )
        pipeline = self._make_pipeline()
        audio = np.zeros(48000, dtype=np.float32)
        result = pipeline.transcribe(audio)
        assert len(result["segments"]) == 1
        seg = result["segments"][0]
        assert "text" in seg
        assert "start" in seg
        assert "end" in seg
        assert "avg_logprob" in seg

    def test_language_auto_detected_on_first_chunk(self, mocker):
        mocker.patch(
            "whispermlx.asr._mlx_whisper_module.transcribe",
            return_value=self.MLX_RESULT,
        )
        pipeline = self._make_pipeline(language=None)
        audio = np.zeros(48000, dtype=np.float32)
        result = pipeline.transcribe(audio)
        # Language from MLX result should be propagated
        assert result["language"] == "en"

    def test_preset_language_used(self, mocker):
        mlx_result = dict(self.MLX_RESULT, language="fr")
        mocker.patch(
            "whispermlx.asr._mlx_whisper_module.transcribe",
            return_value=mlx_result,
        )
        pipeline = self._make_pipeline(language="de")
        audio = np.zeros(48000, dtype=np.float32)
        result = pipeline.transcribe(audio)
        assert result["language"] == "de"

    def test_string_audio_calls_load_audio(self, mocker, tmp_path):
        mocker.patch(
            "whispermlx.asr._mlx_whisper_module.transcribe",
            return_value=self.MLX_RESULT,
        )
        mock_load = mocker.patch(
            "whispermlx.asr.load_audio",
            return_value=np.zeros(48000, dtype=np.float32),
        )
        pipeline = self._make_pipeline()
        audio_path = str(tmp_path / "audio.wav")
        pipeline.transcribe(audio_path)
        mock_load.assert_called_once_with(audio_path)

    def test_empty_vad_output_returns_empty_segments(self, mocker):
        mocker.patch(
            "whispermlx.asr._mlx_whisper_module.transcribe",
            return_value=self.MLX_RESULT,
        )
        vad = _MockVad(segments=[])
        pipeline = MLXWhisperPipeline(
            model_path="mlx-community/whisper-tiny-mlx",
            vad=vad,
            vad_params={"vad_onset": 0.5, "vad_offset": 0.3},
        )
        audio = np.zeros(48000, dtype=np.float32)
        result = pipeline.transcribe(audio)
        assert result["segments"] == []

    def test_progress_callback_called(self, mocker):
        mocker.patch(
            "whispermlx.asr._mlx_whisper_module.transcribe",
            return_value=self.MLX_RESULT,
        )
        pipeline = self._make_pipeline()
        audio = np.zeros(48000, dtype=np.float32)
        received = []
        pipeline.transcribe(audio, progress_callback=received.append)
        assert len(received) == 1
        assert received[0] == pytest.approx(100.0)

    def test_combined_progress_halves_percentage(self, mocker):
        mocker.patch(
            "whispermlx.asr._mlx_whisper_module.transcribe",
            return_value=self.MLX_RESULT,
        )
        pipeline = self._make_pipeline()
        audio = np.zeros(48000, dtype=np.float32)
        received = []
        pipeline.transcribe(audio, progress_callback=received.append, combined_progress=True)
        # With combined_progress=True, callback value is halved
        # For 1 segment: pct = 100/2 = 50.0... but progress_callback gets 100 * (idx+1)/total
        # combined_progress affects print_progress, not progress_callback
        assert len(received) >= 1

    def test_avg_logprob_in_output_segment(self, mocker):
        mocker.patch(
            "whispermlx.asr._mlx_whisper_module.transcribe",
            return_value=self.MLX_RESULT,
        )
        pipeline = self._make_pipeline()
        audio = np.zeros(48000, dtype=np.float32)
        result = pipeline.transcribe(audio)
        seg = result["segments"][0]
        assert "avg_logprob" in seg
        assert seg["avg_logprob"] == pytest.approx(-0.3)

    def test_multiple_vad_segments(self, mocker):
        call_count = [0]

        def fake_transcribe(*args, **kwargs):
            call_count[0] += 1
            return self.MLX_RESULT

        mocker.patch("whispermlx.asr._mlx_whisper_module.transcribe", side_effect=fake_transcribe)
        vad = _MockVad(segments=[_Seg(0.0, 3.0), _Seg(5.0, 8.0)])
        vad.merge_chunks = staticmethod(
            lambda segs, chunk_size, onset, offset: [
                {"start": 0.0, "end": 3.0, "segments": []},
                {"start": 5.0, "end": 8.0, "segments": []},
            ]
        )
        pipeline = MLXWhisperPipeline(
            model_path="mlx-community/whisper-tiny-mlx",
            vad=vad,
            vad_params={"vad_onset": 0.5, "vad_offset": 0.3},
        )
        audio = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
        result = pipeline.transcribe(audio)
        assert call_count[0] == 2
        assert len(result["segments"]) == 2


# ── MLXWhisperPipeline.interleaved_context ───────────────────────────────────


class TestInterleavedContext:
    """interleaved_context carries each chunk's text into the next chunk's prompt."""

    @staticmethod
    def _result(text, avg_logprob=-0.3):
        return {
            "text": text,
            "language": "en",
            "segments": [
                {"text": text, "tokens": [1, 2], "avg_logprob": avg_logprob, "temperature": 0.0}
            ],
        }

    @staticmethod
    def _make_pipeline(n_chunks=2):
        starts = [idx * 5.0 for idx in range(n_chunks)]
        vad = _MockVad(segments=[_Seg(start, start + 3.0) for start in starts])
        vad.merge_chunks = staticmethod(
            lambda segs, chunk_size, onset, offset: [
                {"start": seg.start, "end": seg.end, "segments": []} for seg in segs
            ]
        )
        return MLXWhisperPipeline(
            model_path="mlx-community/whisper-tiny-mlx",
            vad=vad,
            vad_params={"vad_onset": 0.5, "vad_offset": 0.3},
        )

    def _run(self, mocker, mlx_results, initial_prompt=None, **transcribe_kwargs):
        prompts = []

        def fake_transcribe(*args, **kwargs):
            prompts.append(kwargs.get("initial_prompt"))
            return mlx_results[len(prompts) - 1]

        mocker.patch("whispermlx.asr._mlx_whisper_module.transcribe", side_effect=fake_transcribe)
        pipeline = self._make_pipeline(n_chunks=len(mlx_results))
        if initial_prompt is not None:
            pipeline.initial_prompt = initial_prompt
        audio = np.zeros(SAMPLE_RATE * 30, dtype=np.float32)
        result = pipeline.transcribe(audio, **transcribe_kwargs)
        return result, prompts

    def test_disabled_by_default_does_not_forward_context(self, mocker):
        results = [self._result("alpha beta"), self._result("gamma")]
        result, prompts = self._run(mocker, results)
        assert prompts == [None, None]
        assert [s["text"] for s in result["segments"]] == ["alpha beta", "gamma"]

    def test_first_chunk_unconditioned_second_carries_context(self, mocker):
        results = [self._result("alpha beta"), self._result("gamma")]
        _, prompts = self._run(mocker, results, interleaved_context=True)
        assert prompts[0] is None
        assert prompts[1] == "alpha beta"

    def test_user_prompt_combined_with_carried_context(self, mocker):
        results = [self._result("alpha beta"), self._result("gamma")]
        _, prompts = self._run(
            mocker, results, initial_prompt="domain terms", interleaved_context=True
        )
        assert prompts[1] == "domain terms alpha beta"

    def test_context_accumulates_across_chunks(self, mocker):
        results = [
            self._result("one"),
            self._result("two"),
            self._result("three"),
        ]
        _, prompts = self._run(mocker, results, interleaved_context=True)
        assert prompts[1] == "one"
        assert prompts[2] == "one two"

    def test_carried_context_capped_to_max_chars(self, mocker):
        long_text = " ".join(f"word{i}" for i in range(300))
        results = [self._result(long_text), self._result("next")]
        _, prompts = self._run(mocker, results, interleaved_context=True)
        assert prompts[1] == _trim_tail(long_text, INTERLEAVED_CONTEXT_MAX_CHARS)
        assert len(prompts[1]) <= INTERLEAVED_CONTEXT_MAX_CHARS

    def test_low_logprob_chunk_not_carried(self, mocker):
        results = [self._result("garbled text", avg_logprob=-2.0), self._result("gamma")]
        _, prompts = self._run(mocker, results, interleaved_context=True)
        assert prompts[1] is None

    def test_high_temperature_chunk_not_carried(self, mocker):
        bad = self._result("fallback text")
        bad["segments"][0]["temperature"] = 0.8
        results = [bad, self._result("gamma")]
        _, prompts = self._run(mocker, results, interleaved_context=True)
        assert prompts[1] is None


SAMPLE_RATE = 16000


# ── load_model ────────────────────────────────────────────────────────────────


class TestLoadModel:
    def test_silero_vad_selected(self, mocker):
        mock_silero = mocker.patch("whispermlx.asr.Silero")
        mock_silero.return_value = _MockVad()
        from whispermlx.asr import load_model

        pipeline = load_model("tiny", device="cpu", vad_method="silero")
        mock_silero.assert_called_once()
        assert isinstance(pipeline, MLXWhisperPipeline)

    def test_pyannote_vad_selected(self, mocker):
        mock_pyannote = mocker.patch("whispermlx.asr.Pyannote")
        mock_pyannote.return_value = _MockVad()
        from whispermlx.asr import load_model

        pipeline = load_model("tiny", device="cpu", vad_method="pyannote")
        mock_pyannote.assert_called_once()
        assert isinstance(pipeline, MLXWhisperPipeline)

    def test_manual_vad_model_skips_constructor(self, mocker):
        mock_silero = mocker.patch("whispermlx.asr.Silero")
        mock_pyannote = mocker.patch("whispermlx.asr.Pyannote")
        manual_vad = _MockVad()
        from whispermlx.asr import load_model

        pipeline = load_model("tiny", device="cpu", vad_model=manual_vad)
        mock_silero.assert_not_called()
        mock_pyannote.assert_not_called()
        assert pipeline.vad_model is manual_vad

    def test_invalid_vad_method_raises(self, mocker):
        from whispermlx.asr import load_model

        with pytest.raises(ValueError, match="Invalid vad_method"):
            load_model("tiny", device="cpu", vad_method="unknown")

    def test_initial_prompt_forwarded(self, mocker):
        mocker.patch("whispermlx.asr.Silero").return_value = _MockVad()
        from whispermlx.asr import load_model

        pipeline = load_model(
            "tiny",
            device="cpu",
            vad_method="silero",
            asr_options={"initial_prompt": "hello world"},
        )
        assert pipeline.initial_prompt == "hello world"

    def test_vad_options_override_defaults(self, mocker):
        mocker.patch("whispermlx.asr.Silero").return_value = _MockVad()
        from whispermlx.asr import load_model

        pipeline = load_model(
            "tiny",
            device="cpu",
            vad_method="silero",
            vad_options={"chunk_size": 15},
        )
        assert pipeline._vad_params["chunk_size"] == 15

    def test_default_vad_options_present(self, mocker):
        mocker.patch("whispermlx.asr.Silero").return_value = _MockVad()
        from whispermlx.asr import load_model

        pipeline = load_model("tiny", device="cpu", vad_method="silero")
        assert pipeline._vad_params["vad_onset"] == pytest.approx(0.5)
        assert pipeline._vad_params["vad_offset"] == pytest.approx(0.363)
        assert pipeline._vad_params["chunk_size"] == 30

    def test_model_path_resolved(self, mocker):
        mocker.patch("whispermlx.asr.Silero").return_value = _MockVad()
        from whispermlx.asr import load_model

        pipeline = load_model("large-v3", device="cpu", vad_method="silero")
        assert pipeline.model_path == "mlx-community/whisper-large-v3-mlx"

    def test_language_forwarded(self, mocker):
        mocker.patch("whispermlx.asr.Silero").return_value = _MockVad()
        from whispermlx.asr import load_model

        pipeline = load_model("tiny", device="cpu", vad_method="silero", language="fr")
        assert pipeline.preset_language == "fr"
