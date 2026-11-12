# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run CLI
uv run whispermlx audio.mp3 --model large-v3

# Lint + format (via prek — Rust-based pre-commit, same .pre-commit-config.yaml format)
prek run --all-files

# Individual tools
uv run ruff check whispermlx/
uv run ruff format whispermlx/
uv run bandit -c pyproject.toml -r whispermlx/

# Build package
uv build

# Bump version (uses commitizen + uv version provider)
uv run cz bump
```

## Architecture

Three-stage pipeline: **VAD + ASR → Alignment → Diarization**

```
Audio → load_audio() (16kHz) → VAD segments → MLX Whisper → [wav2vec2 alignment] → [pyannote diarization] → output
```

### Key modules

- **`asr.py`** — `MLXWhisperPipeline` + `load_model()`. Iterates VAD segments, calls `mlx_whisper.transcribe()` per chunk. Short model names (e.g. `large-v3`) are mapped to `mlx-community/whisper-*` HF repos via `MLX_MODEL_MAP`.
- **`transcribe.py`** — `transcribe_task()` orchestrates all three stages; explicit `gc.collect()` between stages for memory.
- **`alignment.py`** — `load_align_model()` / `align()`: wav2vec2-based forced alignment for word-level timestamps. Supports 30+ languages. `_mlx_align.py` holds the MLX wav2vec2 CTC encoder used by the default `auto` backend; checkpoints without safetensors fall back to torch.
- **`diarize.py`** — `DiarizationPipeline` (pyannote) + `assign_word_speakers()`. Uses a custom `IntervalTree` (binary search) for O(log n) speaker lookup.
- **`vads/`** — Pluggable VAD via abstract `Vad` base class. Two implementations: `Pyannote` (loads local `assets/pytorch_model.bin`) and `Silero` (torch.hub).
- **`schema.py`** — TypedDict definitions for all data structures (`TranscriptionResult`, `SingleSegment`, `AlignedTranscriptionResult`, etc.).
- **`__init__.py`** — Lazy imports via `importlib` to avoid expensive startup cost.
- **`__main__.py`** — CLI with `argparse`; maps to `transcribe_task()`.

### MLX inference

`mlx-whisper` runs on Apple Silicon GPU automatically — `device` parameter only controls VAD (pyannote). `compute_type`, `device_index`, `threads` are accepted for API compatibility but ignored for inference.

### Dependencies

- `mlx-whisper` — ASR inference (Apple Silicon only)
- `mlx-audio` — MLX wav2vec2 encoder for forced alignment
- `torch` / `torchaudio` — required by pyannote VAD, diarization and the alignment fallback path (CPU wheels via pytorch-cpu index)
- `transformers` — only the alignment fallback path (checkpoints without safetensors weights)
- `pyannote-audio` — VAD + speaker diarization

## Tooling

- **Package manager**: `uv` (always use `uv run`, `uv sync`, `uv build`)
- **Pre-commit runner**: `prek` (https://github.com/j178/prek — Rust drop-in for pre-commit, same `.pre-commit-config.yaml`)
- **Linter/formatter**: `ruff` (isort with `force-single-line`, rules E/F/W/I/B/UP)
- **Security**: `bandit` (skips B101)
- **Versioning**: `commitizen` with conventional commits, `tag_format = "v$version"`
- **CI**: GitHub Actions — `python-compatibility.yml` (macos-latest, Python 3.10–3.14), `build-and-release.yml` (PyPI trusted publishing via OIDC on tag push)
