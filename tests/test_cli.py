"""Tests for whispermlx/__main__.py — CLI argument parsing."""

import sys

import pytest


def _run_cli(monkeypatch, argv, captured):
    """Set sys.argv and run cli(), capturing what gets passed to transcribe_task."""

    def fake_transcribe_task(args, parser):
        captured.update(args)

    monkeypatch.setattr(sys, "argv", argv)

    # Patch transcribe_task before importing cli so the import picks up the mock
    import whispermlx.transcribe as _transcribe_mod

    monkeypatch.setattr(_transcribe_mod, "transcribe_task", fake_transcribe_task)

    # Also patch at the __main__ level since it imports inside the function
    import whispermlx.__main__ as _main_mod
    from whispermlx.__main__ import cli

    monkeypatch.setattr(_main_mod, "cli", cli)

    # Directly call transcribe_task via the import-in-function path
    # by patching at the module level where __main__.cli imports it

    import whispermlx.transcribe

    monkeypatch.setattr(whispermlx.transcribe, "transcribe_task", fake_transcribe_task)
    cli()


@pytest.mark.unit
class TestCLIParsing:
    def test_default_model_is_small(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(monkeypatch, ["whispermlx", str(tmp_path / "audio.wav")], captured)
        assert captured["model"] == "small"

    def test_custom_model(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--model", "large-v3"],
            captured,
        )
        assert captured["model"] == "large-v3"

    def test_language_option(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--language", "en"],
            captured,
        )
        assert captured["language"] == "en"

    def test_diarize_flag(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--diarize"],
            captured,
        )
        assert captured["diarize"] is True

    def test_no_align_flag(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--no_align"],
            captured,
        )
        assert captured["no_align"] is True

    def test_vad_onset_float(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--vad_onset", "0.3"],
            captured,
        )
        assert captured["vad_onset"] == pytest.approx(0.3)

    def test_output_format_srt(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--output_format", "srt"],
            captured,
        )
        assert captured["output_format"] == "srt"

    def test_invalid_output_format_exits(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys, "argv", ["whispermlx", str(tmp_path / "audio.wav"), "--output_format", "mp3"]
        )
        from whispermlx.__main__ import cli

        with pytest.raises(SystemExit) as exc_info:
            cli()
        assert exc_info.value.code == 2

    def test_version_exits(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["whispermlx", "--version"])
        from whispermlx.__main__ import cli

        with pytest.raises(SystemExit) as exc_info:
            cli()
        assert exc_info.value.code == 0

    def test_speaker_embeddings_flag(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--speaker_embeddings"],
            captured,
        )
        assert captured["speaker_embeddings"] is True

    def test_interleaved_context_flag(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--interleaved_context"],
            captured,
        )
        assert captured["interleaved_context"] is True

    def test_default_task_is_transcribe(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(monkeypatch, ["whispermlx", str(tmp_path / "audio.wav")], captured)
        assert captured["task"] == "transcribe"

    def test_translate_task(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--task", "translate"],
            captured,
        )
        assert captured["task"] == "translate"

    def test_default_vad_method_is_pyannote(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(monkeypatch, ["whispermlx", str(tmp_path / "audio.wav")], captured)
        assert captured["vad_method"] == "pyannote"

    def test_silero_vad_method(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            ["whispermlx", str(tmp_path / "audio.wav"), "--vad_method", "silero"],
            captured,
        )
        assert captured["vad_method"] == "silero"

    def test_multiple_audio_files(self, monkeypatch, tmp_path):
        captured = {}
        _run_cli(
            monkeypatch,
            [
                "whispermlx",
                str(tmp_path / "a.wav"),
                str(tmp_path / "b.wav"),
            ],
            captured,
        )
        assert len(captured["audio"]) == 2
