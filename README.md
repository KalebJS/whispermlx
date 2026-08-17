<h1 align="center">whisper(ml)x</h1>

<p align="center">
  Fast, accurate speech recognition on Apple Silicon — powered by <a href="https://github.com/ml-explore/mlx">MLX</a>.
</p>

<p align="center">
  <a href="https://whispermlx.docs.kalebjs.dev">Documentation</a>
</p>

A fork of [WhisperX](https://github.com/m-bain/whisperX) with the inference backend replaced by [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), running natively on Apple Silicon via MLX. Word-level timestamps, speaker diarization, and VAD are all retained.

- ⚡️ MLX inference — runs on Apple Silicon GPU via unified memory
- 🎯 Word-level timestamps via wav2vec2 forced alignment
- 👥 Speaker diarization via [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- 🗣️ VAD preprocessing via pyannote or silero

## Installation

```bash
pip install whispermlx
```

Or with uv:

```bash
uv add whispermlx
```

## Usage

### CLI

```bash
# Auto-downloads mlx-community/whisper-large-v3-mlx on first run
whispermlx audio.mp3 --model large-v3

# With speaker diarization
whispermlx audio.mp3 --model large-v3 --diarize --hf_token YOUR_TOKEN

# Use any mlx-community model directly
whispermlx audio.mp3 --model mlx-community/whisper-large-v3-turbo
```

### Python

```python
import whispermlx

# Short name — auto-maps to mlx-community/whisper-large-v3-mlx
model = whispermlx.load_model("large-v3", device="cpu")
result = model.transcribe("audio.mp3")
print(result["segments"])

# With alignment
model_a, metadata = whispermlx.load_align_model(language_code=result["language"], device="cpu")
result = whispermlx.align(result["segments"], model_a, metadata, "audio.mp3", device="cpu")

# With diarization
from whispermlx.diarize import DiarizationPipeline
diarize_model = DiarizationPipeline(token="YOUR_HF_TOKEN", device="cpu")
diarize_segments = diarize_model("audio.mp3")
result = whispermlx.assign_word_speakers(diarize_segments, result)
```

## Model Names

Short names are automatically mapped to their `mlx-community` equivalents. Full HF repo IDs also work.

| Short name | HF repo |
|---|---|
| `tiny`, `base`, `small`, `medium` | `mlx-community/whisper-{name}-mlx` |
| `large-v3` | `mlx-community/whisper-large-v3-mlx` |
| `large-v3-turbo` / `turbo` | `mlx-community/whisper-large-v3-turbo` |

## Comparison with WhisperX

whispermlx tracks upstream [WhisperX](https://github.com/m-bain/whisperX)'s pipeline and API, with the inference backend swapped to MLX. Everything downstream of ASR (alignment, diarization, output formats) is unchanged.

### Feature parity

| Feature | WhisperX | whispermlx | Notes |
|---|---|---|---|
| VAD preprocessing | ✅ pyannote / silero | ✅ pyannote / silero | Identical |
| Word-level timestamps | ✅ wav2vec2 alignment | ✅ wav2vec2 alignment | Identical; adds Indonesian (`id`) model |
| Speaker diarization | ✅ pyannote-audio | ✅ pyannote-audio | Identical |
| Output formats | srt, vtt, txt, tsv, json, aud | srt, vtt, txt, tsv, json, aud | Identical |
| CLI flags | Full set | Full set | Identical; adds `--log-level` |
| Python API | `load_model`, `transcribe`, `align`, `assign_word_speakers` | Same signatures | Drop-in compatible |
| Batched inference | ✅ faster-whisper batching | ❌ per-segment | `batch_size` accepted but unused |
| CUDA / NVIDIA GPU | ✅ | ❌ | Apple Silicon only |
| Apple Silicon GPU | ❌ (CPU only) | ✅ MLX unified memory | Native, no CUDA |

### API compatibility

| Parameter | WhisperX | whispermlx |
|---|---|---|
| `device` | Controls ASR + PyTorch models | Controls VAD/alignment/diarization only; MLX inference auto-uses GPU |
| `compute_type` | float16 / float32 / int8 | Accepted, ignored |
| `device_index`, `threads`, `download_root`, `local_files_only`, `use_auth_token` | Used | Accepted for compatibility, ignored |
| `asr_options` | Full faster-whisper options | Only `initial_prompt` is used |
| `--model` | faster-whisper model names | Short names map to `mlx-community` repos; full HF repo IDs also work |

### Dependencies

| Purpose | WhisperX | whispermlx |
|---|---|---|
| ASR backend | ctranslate2 + faster-whisper | mlx-whisper |
| Alignment | transformers (wav2vec2) | transformers (wav2vec2) |
| VAD + diarization | pyannote-audio | pyannote-audio |
| Torch | CUDA/CPU wheels | CPU wheels only |
| Extra | triton, torchcodec | numba, tqdm |

## Speaker Diarization

Requires a [Hugging Face access token](https://huggingface.co/settings/tokens) and acceptance of the [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) model agreement.

## Acknowledgements

Built on top of [WhisperX](https://github.com/m-bain/whisperX) by Max Bain et al., [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), [pyannote-audio](https://github.com/pyannote/pyannote-audio), and [OpenAI Whisper](https://github.com/openai/whisper).

```bibtex
@article{bain2022whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  journal={INTERSPEECH 2023},
  year={2023}
}
```
