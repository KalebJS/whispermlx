import contextlib
import io

import mlx_whisper as _mlx_whisper_module
import numpy as np
from tqdm import tqdm

from whispermlx.audio import SAMPLE_RATE
from whispermlx.audio import load_audio
from whispermlx.log_utils import get_logger
from whispermlx.schema import ProgressCallback
from whispermlx.schema import SingleSegment
from whispermlx.schema import TranscriptionResult
from whispermlx.vads import Pyannote
from whispermlx.vads import Silero
from whispermlx.vads import Vad

logger = get_logger(__name__)

MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large": "mlx-community/whisper-large-mlx",
    "large-v1": "mlx-community/whisper-large-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def _resolve_mlx_model(whisper_arch: str) -> str:
    """Map a short name or full HF repo ID to an mlx-community repo."""
    if "/" in whisper_arch:
        return whisper_arch
    if whisper_arch in MLX_MODEL_MAP:
        return MLX_MODEL_MAP[whisper_arch]
    logger.warning(
        f"Unknown model '{whisper_arch}'. "
        f"Pass a full HF repo ID or one of: {list(MLX_MODEL_MAP.keys())}"
    )
    return whisper_arch


def _compute_avg_logprob(mlx_segments: list) -> float:
    if not mlx_segments:
        return 0.0
    total, weighted = 0, 0.0
    for seg in mlx_segments:
        n = len(seg.get("tokens", [])) or 1
        weighted += seg.get("avg_logprob", 0.0) * n
        total += n
    return weighted / total if total else 0.0


class MLXWhisperPipeline:
    """WhisperX transcription pipeline using mlx-whisper on Apple Silicon."""

    def __init__(
        self,
        model_path: str,
        vad,
        vad_params: dict,
        language: str | None = None,
        task: str = "transcribe",
        initial_prompt: str | None = None,
    ):
        self.model_path = model_path
        self.vad_model = vad
        self._vad_params = vad_params
        self.preset_language = language
        self.task = task
        self.initial_prompt = initial_prompt

    def transcribe(
        self,
        audio: str | np.ndarray,
        batch_size: int | None = None,
        num_workers: int = 0,
        language: str | None = None,
        task: str | None = None,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        verbose: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        effective_language = language or self.preset_language
        effective_task = task or self.task

        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        segments: list[SingleSegment] = []
        total_segments = len(vad_segments)

        pbar = tqdm(total=total_segments, desc="Transcribing", unit="seg")
        for idx, vad_seg in enumerate(vad_segments):
            f1 = int(vad_seg["start"] * SAMPLE_RATE)
            f2 = int(vad_seg["end"] * SAMPLE_RATE)
            audio_chunk = audio[f1:f2]

            with contextlib.redirect_stderr(io.StringIO()):
                mlx_result = _mlx_whisper_module.transcribe(
                    audio_chunk,
                    path_or_hf_repo=self.model_path,
                    language=effective_language,
                    task=effective_task,
                    verbose=False,
                    initial_prompt=self.initial_prompt,
                    word_timestamps=False,
                )

            if effective_language is None and idx == 0:
                effective_language = mlx_result.get("language")

            chunk_text = mlx_result.get("text", "").strip()
            avg_logprob = _compute_avg_logprob(mlx_result.get("segments", []))

            pbar.update(1)
            if verbose:
                tqdm.write(
                    f"[{round(vad_seg['start'], 3)} --> {round(vad_seg['end'], 3)}] {chunk_text}"
                )
            if print_progress:
                base = ((idx + 1) / total_segments) * 100
                pct = base / 2 if combined_progress else base
                tqdm.write(f"Progress: {pct:.2f}%...")
            if progress_callback is not None:
                progress_callback(((idx + 1) / total_segments) * 100)

            segments.append(
                {
                    "text": chunk_text,
                    "start": round(vad_seg["start"], 3),
                    "end": round(vad_seg["end"], 3),
                    "avg_logprob": avg_logprob,
                }
            )

        pbar.close()
        return {"segments": segments, "language": effective_language or "en"}


def load_model(
    whisper_arch: str,
    device: str,
    device_index: int = 0,
    compute_type: str = "default",
    asr_options: dict | None = None,
    language: str | None = None,
    vad_model: Vad | None = None,
    vad_method: str | None = "pyannote",
    vad_options: dict | None = None,
    model=None,
    task: str = "transcribe",
    download_root: str | None = None,
    local_files_only: bool = False,
    threads: int = 4,
    use_auth_token=None,
) -> MLXWhisperPipeline:
    """Load a Whisper model for inference using the MLX backend.

    Args:
        whisper_arch: Short model name (e.g. "large-v3", "small") or a full
            HuggingFace repo ID (e.g. "mlx-community/whisper-large-v3-mlx").
        device: Device for the VAD model ("cpu" recommended on Apple Silicon;
            MLX inference uses the GPU automatically).
        device_index: Kept for API compatibility, unused.
        compute_type: Kept for API compatibility, ignored.
        asr_options: Dict of ASR options; only "initial_prompt" is used.
        language: Language code, or None for auto-detection.
        vad_model: Pre-instantiated VAD model (overrides vad_method).
        vad_method: "pyannote" or "silero".
        vad_options: VAD configuration overrides.
        model: Kept for API compatibility, ignored.
        task: "transcribe" or "translate".
        download_root: Kept for API compatibility, ignored (HF cache used).
        local_files_only: Kept for API compatibility, ignored.
        threads: Kept for API compatibility, ignored.
        use_auth_token: Kept for API compatibility, ignored.
    Returns:
        An MLXWhisperPipeline ready for transcription.
    """
    model_path = _resolve_mlx_model(whisper_arch)
    logger.info(f"Loading MLX Whisper model: {model_path}")

    initial_prompt = (asr_options or {}).get("initial_prompt")

    default_vad_options = {
        "chunk_size": 30,
        "vad_onset": 0.500,
        "vad_offset": 0.363,
    }
    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is not None:
        logger.info("Using manually assigned vad_model.")
        resolved_vad = vad_model
    else:
        if vad_method == "silero":
            resolved_vad = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            import torch

            device_vad = f"cuda:{device_index}" if device == "cuda" else device
            resolved_vad = Pyannote(torch.device(device_vad), token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    return MLXWhisperPipeline(
        model_path=model_path,
        vad=resolved_vad,
        vad_params=default_vad_options,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
    )
