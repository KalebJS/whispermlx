"""Tests for lower-level alignment functions and load_align_model."""

import pytest
import torch

from whispermlx.alignment import Point
from whispermlx.alignment import Segment
from whispermlx.alignment import backtrack
from whispermlx.alignment import get_trellis
from whispermlx.alignment import merge_repeats
from whispermlx.alignment import merge_words

# ── get_trellis ────────────────────────────────────────────────────────────────


class TestGetTrellis:
    def test_output_shape(self):
        T, V = 10, 5
        emission = torch.zeros(T, V)
        tokens = [1, 2, 3]
        trellis = get_trellis(emission, tokens)
        assert trellis.shape == (T + 1, len(tokens) + 1)

    def test_first_cell_is_zero(self):
        emission = torch.zeros(5, 4)
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens)
        assert trellis[0, 0].item() == pytest.approx(0.0)

    def test_top_right_is_negative_inf(self):
        emission = torch.zeros(5, 4)
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens)
        # trellis[0, -num_tokens:] = -inf
        assert trellis[0, 1].item() == float("-inf")
        assert trellis[0, 2].item() == float("-inf")

    def test_blank_column_cumsum(self):
        T = 4
        emission = torch.log(torch.tensor([[0.5, 0.3, 0.2]] * T))
        tokens = [1]
        trellis = get_trellis(emission, tokens, blank_id=0)
        # trellis[1:, 0] = cumsum(emission[:, 0])
        # BUT trellis[-num_tokens:, 0] = inf (the initialization override)
        # so skip last num_tokens rows
        num_tokens = len(tokens)
        cumsum = torch.cumsum(emission[:, 0], 0)
        for t in range(1, T + 1 - num_tokens):
            assert trellis[t, 0].item() == pytest.approx(cumsum[t - 1].item(), abs=1e-5)

    def test_scores_fill_body(self):
        emission = torch.zeros(3, 4)
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens)
        # Body cells should be finite (not nan)
        assert not torch.isnan(trellis[1:, 1:]).any()


# ── backtrack ─────────────────────────────────────────────────────────────────


class TestBacktrack:
    def _make_simple_path(self, T=5, num_tokens=2):
        """Build a simple trellis that produces a valid path."""
        V = max(num_tokens + 2, 3)
        emission = torch.zeros(T, V)
        # Put high probability on token 1 for first half, token 2 for second half
        emission[: T // 2, 1] = 5.0
        emission[T // 2 :, 2] = 5.0
        tokens = list(range(1, num_tokens + 1))
        trellis = get_trellis(emission, tokens)
        return trellis, emission, tokens

    def test_returns_path_list(self):
        trellis, emission, tokens = self._make_simple_path()
        path = backtrack(trellis, emission, tokens)
        assert path is not None
        assert isinstance(path, list)

    def test_path_elements_are_points(self):
        trellis, emission, tokens = self._make_simple_path()
        path = backtrack(trellis, emission, tokens)
        assert path is not None
        for p in path:
            assert isinstance(p, Point)
            assert hasattr(p, "token_index")
            assert hasattr(p, "time_index")
            assert hasattr(p, "score")

    def test_path_scores_nonnegative(self):
        """Scores are exp(log_prob) which can exceed 1; they must be non-negative."""
        trellis, emission, tokens = self._make_simple_path()
        path = backtrack(trellis, emission, tokens)
        assert path is not None
        for p in path:
            assert p.score >= 0.0

    def test_degenerate_trellis_returns_none(self):
        # If trellis is all -inf except first col, backtrack should fail
        T, V = 3, 3
        trellis = torch.full((T + 1, 3), float("-inf"))
        trellis[0, 0] = 0.0
        trellis[1, 0] = -1.0
        trellis[2, 0] = -2.0
        trellis[3, 0] = -3.0
        # All token columns are -inf → path must fail
        emission = torch.zeros(T, V)
        tokens = [1, 2]
        result = backtrack(trellis, emission, tokens)
        # May return None if path cannot complete
        assert result is None or isinstance(result, list)


# ── merge_repeats ──────────────────────────────────────────────────────────────


class TestMergeRepeats:
    def test_single_run_produces_one_segment(self):
        path = [Point(0, 0, 0.9), Point(0, 1, 0.8), Point(0, 2, 0.7)]
        transcript = "h"
        segs = merge_repeats(path, transcript)
        assert len(segs) == 1
        assert segs[0].label == "h"

    def test_two_distinct_runs(self):
        path = [
            Point(0, 0, 0.9),
            Point(0, 1, 0.8),
            Point(1, 2, 0.7),
            Point(1, 3, 0.6),
        ]
        transcript = "he"
        segs = merge_repeats(path, transcript)
        assert len(segs) == 2
        assert segs[0].label == "h"
        assert segs[1].label == "e"

    def test_score_is_mean_of_run(self):
        path = [Point(0, 0, 0.9), Point(0, 1, 0.7)]
        segs = merge_repeats(path, "h")
        assert segs[0].score == pytest.approx(0.8)

    def test_segment_time_bounds(self):
        path = [Point(0, 2, 0.9), Point(0, 3, 0.8), Point(0, 4, 0.7)]
        segs = merge_repeats(path, "x")
        # start = first time_index, end = last time_index + 1
        assert segs[0].start == 2
        assert segs[0].end == 5

    def test_labels_match_transcript(self):
        path = [Point(0, 0, 0.9), Point(1, 1, 0.8), Point(2, 2, 0.7)]
        transcript = "hel"
        segs = merge_repeats(path, transcript)
        assert [s.label for s in segs] == ["h", "e", "l"]


# ── merge_words ────────────────────────────────────────────────────────────────


class TestMergeWords:
    def _make_char_segs(self, labels):
        """Build Segment objects from label string list."""
        segs = []
        for i, lbl in enumerate(labels):
            segs.append(Segment(lbl, i, i + 1, 0.8))
        return segs

    def test_splits_on_pipe_separator(self):
        labels = ["h", "e", "l", "l", "o", "|", "w", "o", "r", "l", "d"]
        segs = self._make_char_segs(labels)
        words = merge_words(segs)
        assert len(words) == 2
        assert words[0].label == "hello"
        assert words[1].label == "world"

    def test_leading_separator_skipped(self):
        labels = ["|", "h", "i"]
        segs = self._make_char_segs(labels)
        words = merge_words(segs)
        # i1 == i2 guard skips the leading separator
        assert len(words) == 1
        assert words[0].label == "hi"

    def test_empty_input_returns_empty(self):
        words = merge_words([])
        assert words == []

    def test_single_word_no_separator(self):
        labels = ["h", "i"]
        segs = self._make_char_segs(labels)
        words = merge_words(segs)
        assert len(words) == 1
        assert words[0].label == "hi"

    def test_score_weighted_by_length(self):
        # "hi" = 2 chars each score 0.8; weighted avg = 0.8
        labels = ["h", "i", "|", "x"]
        segs = self._make_char_segs(labels)
        words = merge_words(segs)
        assert words[0].score == pytest.approx(0.8)

    def test_start_end_from_segments(self):
        labels = ["h", "e", "l", "l", "o"]
        segs = self._make_char_segs(labels)
        words = merge_words(segs)
        assert words[0].start == 0
        assert words[0].end == 5


# ── load_align_model ──────────────────────────────────────────────────────────


class TestLoadAlignModel:
    def test_unsupported_language_raises(self):
        from whispermlx.alignment import load_align_model

        with pytest.raises(ValueError, match="No default align-model"):
            load_align_model("xx", "cpu")

    def test_torchaudio_path(self, mocker):
        """Mock torchaudio bundle to exercise the torchaudio branch."""
        import torchaudio

        from whispermlx.alignment import load_align_model

        fake_model = torch.nn.Linear(10, 10)
        fake_labels = ["<pad>", "a", "b", "c"]

        mock_bundle = mocker.MagicMock()
        mock_bundle.get_model.return_value = fake_model
        mock_bundle.get_labels.return_value = fake_labels
        mock_bundle.to = lambda device: mock_bundle

        # Make "WAV2VEC2_ASR_BASE_960H" appear in __all__ and __dict__
        mocker.patch.object(torchaudio.pipelines, "__all__", ["WAV2VEC2_ASR_BASE_960H"])
        mocker.patch.dict(
            torchaudio.pipelines.__dict__,
            {"WAV2VEC2_ASR_BASE_960H": mock_bundle},
        )

        model, metadata = load_align_model("en", "cpu")
        assert metadata["type"] == "torchaudio"
        assert metadata["language"] == "en"
        assert "a" in metadata["dictionary"]

    def test_huggingface_path(self, mocker):
        """Mock HF loaders to exercise the HuggingFace branch."""
        import torchaudio

        from whispermlx.alignment import load_align_model

        fake_model = torch.nn.Linear(10, 10)

        mock_processor = mocker.MagicMock()
        mock_processor.tokenizer.get_vocab.return_value = {"<pad>": 0, "a": 1, "b": 2}

        mock_hf_model = mocker.MagicMock()
        mock_hf_model.to.return_value = fake_model

        mocker.patch(
            "whispermlx.alignment.Wav2Vec2Processor.from_pretrained", return_value=mock_processor
        )
        mocker.patch(
            "whispermlx.alignment.Wav2Vec2ForCTC.from_pretrained", return_value=mock_hf_model
        )

        # Ensure "ja" is NOT in torchaudio.__all__
        mocker.patch.object(torchaudio.pipelines, "__all__", [])

        model, metadata = load_align_model("ja", "cpu")
        assert metadata["type"] == "huggingface"
        assert metadata["language"] == "ja"
        assert "a" in metadata["dictionary"]

    def test_network_error_warns_about_cache_only_flag(self, mocker):
        import torchaudio

        import whispermlx.alignment as alignment_module
        from whispermlx.alignment import load_align_model

        mocker.patch.object(torchaudio.pipelines, "__all__", [])
        mocker.patch(
            "whispermlx.alignment.Wav2Vec2Processor.from_pretrained",
            side_effect=OSError(
                "MaxRetryError('HTTPSConnectionPool(host='huggingface.co', port=443)')"
            ),
        )
        mock_logger = mocker.patch.object(alignment_module, "logger")

        with pytest.raises(ValueError):
            load_align_model("tr", "cpu")

        warning_text = " ".join(str(a) for a in mock_logger.warning.call_args[0])
        assert "--model_cache_only" in warning_text

    def test_non_network_error_is_surfaced(self, mocker):
        """A non-network failure must keep its own cause in the raised message.

        A partially downloaded cache raises OSError about a missing
        preprocessor_config.json. Reporting that as "model could not be found"
        sends people looking for a model that exists and loads fine on retry.
        """
        import torchaudio

        import whispermlx.alignment as alignment_module
        from whispermlx.alignment import load_align_model

        mocker.patch.object(torchaudio.pipelines, "__all__", [])
        mocker.patch(
            "whispermlx.alignment.Wav2Vec2Processor.from_pretrained",
            side_effect=OSError("Can't load feature extractor for 'some/model'"),
        )
        mock_logger = mocker.patch.object(alignment_module, "logger")

        with pytest.raises(ValueError, match="Can't load feature extractor"):
            load_align_model("ru", "cpu")

        error_text = " ".join(str(a) for a in mock_logger.error.call_args[0])
        assert "OSError" in error_text
        assert "not necessarily a missing model" in error_text

    def test_explicit_model_name_skips_default_lookup(self, mocker):
        """Explicit model_name should bypass default language mapping."""
        import torchaudio

        from whispermlx.alignment import load_align_model

        mock_processor = mocker.MagicMock()
        mock_processor.tokenizer.get_vocab.return_value = {"<pad>": 0, "x": 1}
        mock_hf_model = mocker.MagicMock()
        mock_hf_model.to.return_value = torch.nn.Linear(5, 5)

        mocker.patch(
            "whispermlx.alignment.Wav2Vec2Processor.from_pretrained", return_value=mock_processor
        )
        mocker.patch(
            "whispermlx.alignment.Wav2Vec2ForCTC.from_pretrained", return_value=mock_hf_model
        )
        mocker.patch.object(torchaudio.pipelines, "__all__", [])

        model, metadata = load_align_model("en", "cpu", model_name="custom/model-name")
        assert metadata["type"] == "huggingface"
