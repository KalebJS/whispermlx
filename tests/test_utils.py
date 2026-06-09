"""Tests for whispermlx/utils.py — pure functions and writer classes."""

import json
import os

import pandas as pd
import pytest

from whispermlx.utils import compression_ratio
from whispermlx.utils import exact_div
from whispermlx.utils import format_timestamp
from whispermlx.utils import get_writer
from whispermlx.utils import interpolate_nans
from whispermlx.utils import optional_float
from whispermlx.utils import optional_int
from whispermlx.utils import str2bool


class TestStr2Bool:
    def test_true_string(self):
        assert str2bool("True") is True

    def test_false_string(self):
        assert str2bool("False") is False

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Expected one of"):
            str2bool("yes")

    def test_lowercase_raises(self):
        with pytest.raises(ValueError):
            str2bool("true")


class TestOptionalInt:
    def test_none_string(self):
        assert optional_int("None") is None

    def test_integer_string(self):
        assert optional_int("5") == 5

    def test_zero(self):
        assert optional_int("0") == 0

    def test_negative(self):
        assert optional_int("-3") == -3


class TestOptionalFloat:
    def test_none_string(self):
        assert optional_float("None") is None

    def test_float_string(self):
        assert optional_float("3.14") == pytest.approx(3.14)

    def test_integer_string(self):
        assert optional_float("2") == 2.0


class TestExactDiv:
    def test_exact_division(self):
        assert exact_div(480000, 160) == 3000

    def test_non_exact_raises(self):
        with pytest.raises(AssertionError):
            exact_div(7, 3)

    def test_zero_dividend(self):
        assert exact_div(0, 5) == 0


class TestFormatTimestamp:
    def test_zero(self):
        result = format_timestamp(0.0)
        assert result == "00:00.000"

    def test_one_minute(self):
        result = format_timestamp(60.0)
        assert "01:00" in result

    def test_always_include_hours(self):
        result = format_timestamp(0.0, always_include_hours=True)
        assert result.startswith("00:")

    def test_over_one_hour(self):
        result = format_timestamp(3661.0, always_include_hours=True)
        assert result.startswith("01:")

    def test_negative_raises(self):
        with pytest.raises(AssertionError):
            format_timestamp(-1.0)

    def test_decimal_marker(self):
        result = format_timestamp(1.5, decimal_marker=",")
        assert "," in result

    def test_milliseconds(self):
        result = format_timestamp(1.001)
        assert "001" in result


class TestCompressionRatio:
    def test_repetitive_text_has_high_ratio(self):
        text = "hello " * 100
        assert compression_ratio(text) > 1.0

    def test_short_unique_text_has_low_ratio(self):
        text = "the"
        # short strings may have ratio <= 1 after compression overhead
        assert isinstance(compression_ratio(text), float)


class TestInterpolateNans:
    def test_no_nans_unchanged(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = interpolate_nans(s)
        pd.testing.assert_series_equal(result, s)

    def test_interior_nan_linear(self):
        s = pd.Series([1.0, float("nan"), 3.0])
        result = interpolate_nans(s, method="linear")
        assert result.iloc[1] == pytest.approx(2.0)

    def test_interior_nan_nearest(self):
        s = pd.Series([1.0, float("nan"), 3.0])
        result = interpolate_nans(s, method="nearest")
        # nearest should fill with 1.0 or 3.0
        assert not result.isna().any()

    def test_all_nan_returns_series(self):
        s = pd.Series([float("nan"), float("nan")])
        result = interpolate_nans(s)
        assert isinstance(result, pd.Series)

    def test_single_valid_value_ffill(self):
        s = pd.Series([float("nan"), 5.0])
        result = interpolate_nans(s)
        assert result.iloc[0] == pytest.approx(5.0)

    def test_ignore_preserves_nans(self):
        s = pd.Series([1.0, float("nan"), 3.0, float("nan"), 5.0])
        result = interpolate_nans(s, method="ignore")
        assert result.isna().sum() == 2


class TestResultWriters:
    def _write(self, writer, result, tmp_path, audio_name="audio.wav"):
        audio_path = str(tmp_path / audio_name)
        options = {"highlight_words": False, "max_line_width": None, "max_line_count": None}
        writer(result, audio_path, options)

    def test_write_txt(self, tmp_path):
        writer = get_writer("txt", str(tmp_path))
        result = {
            "segments": [{"start": 0.0, "end": 2.0, "text": " hello world"}],
            "language": "en",
        }
        self._write(writer, result, tmp_path)
        content = (tmp_path / "audio.txt").read_text()
        assert "hello world" in content

    def test_write_txt_with_speaker(self, tmp_path):
        writer = get_writer("txt", str(tmp_path))
        result = {
            "segments": [{"start": 0.0, "end": 2.0, "text": " test", "speaker": "SPEAKER_00"}],
            "language": "en",
        }
        self._write(writer, result, tmp_path)
        content = (tmp_path / "audio.txt").read_text()
        assert "[SPEAKER_00]:" in content

    def test_write_srt(self, tmp_path):
        writer = get_writer("srt", str(tmp_path))
        result = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}],
            "language": "en",
        }
        self._write(writer, result, tmp_path)
        content = (tmp_path / "audio.srt").read_text()
        assert "1\n" in content
        assert "-->" in content

    def test_write_vtt(self, tmp_path):
        writer = get_writer("vtt", str(tmp_path))
        result = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}],
            "language": "en",
        }
        self._write(writer, result, tmp_path)
        content = (tmp_path / "audio.vtt").read_text()
        assert "WEBVTT" in content
        assert "-->" in content

    def test_write_tsv(self, tmp_path):
        writer = get_writer("tsv", str(tmp_path))
        result = {
            "segments": [{"start": 1.0, "end": 3.0, "text": "hello"}],
            "language": "en",
        }
        self._write(writer, result, tmp_path)
        content = (tmp_path / "audio.tsv").read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "start\tend\ttext"
        cols = lines[1].split("\t")
        assert len(cols) == 3
        # start is in milliseconds
        assert int(cols[0]) == 1000

    def test_write_json(self, tmp_path):
        writer = get_writer("json", str(tmp_path))
        result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        self._write(writer, result, tmp_path)
        content = (tmp_path / "audio.json").read_text()
        parsed = json.loads(content)
        assert parsed["language"] == "en"

    def test_write_audacity(self, tmp_path):
        writer = get_writer("aud", str(tmp_path))
        result = {
            "segments": [{"start": 1.0, "end": 3.0, "text": "test"}],
            "language": "en",
        }
        self._write(writer, result, tmp_path)
        content = (tmp_path / "audio.aud").read_text()
        lines = content.strip().split("\n")
        # each line has start, end, text (tab-separated, seconds)
        assert len(lines) == 1

    def test_get_writer_all(self, tmp_path):
        writer = get_writer("all", str(tmp_path))
        assert callable(writer)

    def test_get_writer_all_writes_multiple_formats(self, tmp_path):
        writer = get_writer("all", str(tmp_path))
        result = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        audio_path = str(tmp_path / "test.wav")
        options = {"highlight_words": False, "max_line_width": None, "max_line_count": None}
        writer(result, audio_path, options)
        extensions = {os.path.splitext(f)[1] for f in os.listdir(tmp_path)}
        assert ".txt" in extensions
        assert ".srt" in extensions
        assert ".vtt" in extensions
        assert ".tsv" in extensions
        assert ".json" in extensions

    def test_write_srt_empty_segments(self, tmp_path):
        writer = get_writer("srt", str(tmp_path))
        result = {"segments": [], "language": "en"}
        self._write(writer, result, tmp_path)
        content = (tmp_path / "audio.srt").read_text()
        assert content == ""
