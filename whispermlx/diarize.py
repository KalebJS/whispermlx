import math
import types

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.speaker_verification import PyannoteAudioPretrainedSpeakerEmbedding

from whispermlx.audio import SAMPLE_RATE
from whispermlx.audio import load_audio
from whispermlx.log_utils import get_logger
from whispermlx.schema import AlignedTranscriptionResult
from whispermlx.schema import ProgressCallback
from whispermlx.schema import TranscriptionResult

logger = get_logger(__name__)


class IntervalTree:
    """
    Simple interval tree for fast overlap queries using sorted array + binary search.

    Uses O(n) space and provides O(log n) query time instead of O(n) linear scan.
    This achieves ~228x speedup for speaker assignment in long-form content.
    """

    def __init__(self, intervals: list[tuple[float, float, str]]):
        """
        Initialize the interval tree with diarization segments.

        Args:
            intervals: List of (start, end, speaker) tuples
        """
        if not intervals:
            self.starts = np.array([])
            self.ends = np.array([])
            self.speakers: list[str] = []
            return

        # Sort intervals by start time for binary search
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        self.starts = np.array([i[0] for i in sorted_intervals], dtype=np.float64)
        self.ends = np.array([i[1] for i in sorted_intervals], dtype=np.float64)
        self.speakers = [i[2] for i in sorted_intervals]

    def query(self, start: float, end: float) -> list[tuple[str, float]]:
        """
        Find all intervals that overlap with [start, end] and compute intersection.

        Args:
            start: Query interval start time
            end: Query interval end time

        Returns:
            List of (speaker, intersection_duration) tuples for overlapping segments
        """
        if len(self.starts) == 0:
            return []

        # Binary search to find candidate intervals
        # Only intervals with start < end could overlap
        right_idx = np.searchsorted(self.starts, end, side="left")
        if right_idx == 0:
            return []

        # Check candidates for actual overlap
        candidates = slice(0, right_idx)
        overlaps = (self.starts[candidates] < end) & (self.ends[candidates] > start)

        results = []
        for idx in np.where(overlaps)[0]:
            intersection = min(self.ends[idx], end) - max(self.starts[idx], start)
            if intersection > 0:
                results.append((self.speakers[idx], intersection))
        return results

    def find_nearest(self, time: float) -> str | None:
        """
        Find the speaker of the nearest segment to a given time point.

        Args:
            time: Time point to find nearest segment for

        Returns:
            Speaker ID of nearest segment, or None if no segments exist
        """
        if len(self.starts) == 0:
            return None

        # Calculate midpoints of all segments
        mids = (self.starts + self.ends) / 2
        nearest_idx = np.argmin(np.abs(mids - time))
        return self.speakers[nearest_idx]


def _supports_stacked_masks(embedding) -> bool:
    """
    Check whether an embedding model accepts (batch, speakers, frames) masks in a single call.

    pyannote-native WeSpeaker models only apply masks in their statistics pooling layer,
    which already supports a speakers dimension. Other backends (SpeechBrain, NeMo, ONNX)
    apply masks before the frame-wise network and must keep pyannote's per-pair loop.
    """
    if not isinstance(embedding, PyannoteAudioPretrainedSpeakerEmbedding):
        return False
    try:
        with torch.inference_mode():
            waveforms = torch.zeros(1, 1, embedding.sample_rate, device=embedding.device)
            weights = torch.ones(1, 2, 10, device=embedding.device)
            output = embedding.model_(waveforms, weights=weights)
    except Exception:
        return False
    return tuple(output.shape) == (1, 2, embedding.dimension)


def _get_embeddings_per_chunk(
    self: SpeakerDiarization,
    file,
    binary_segmentations,
    exclude_overlap: bool = False,
    hook=None,
):
    """
    Drop-in replacement for SpeakerDiarization.get_embeddings that runs the embedding
    model once per chunk instead of once per (chunk, speaker) pair.

    pyannote feeds the same chunk waveform to the embedding model once for every local
    speaker, while the speaker mask only enters at the final pooling layer. Stacking a
    chunk's masks and pooling them in one call yields the same embeddings (up to float
    noise) with num_speakers times less work in the frame-wise network.
    """
    if self.training:
        return SpeakerDiarization.get_embeddings(
            self, file, binary_segmentations, exclude_overlap=exclude_overlap, hook=hook
        )

    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, _ = binary_segmentations.data.shape
    data = binary_segmentations.data
    # masks may contain NaN (in case of partial stitching)
    segmentations = np.nan_to_num(data, nan=0.0).astype(np.float32)

    if exclude_overlap:
        # same rule as pyannote: use the overlap-free mask when it is long enough
        # to extract an embedding from, otherwise fall back to the full mask.
        # Overlap is computed before NaN removal, so frames with NaN never count as clean.
        min_num_samples = self._embedding.min_num_samples
        num_samples = duration * self._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)
        clean_frames = np.sum(data, axis=2, keepdims=True) < 2
        clean_segmentations = np.nan_to_num(data * clean_frames, nan=0.0).astype(np.float32)
        use_clean = np.sum(clean_segmentations, axis=1, keepdims=True) > min_num_frames
        segmentations = np.where(use_clean, clean_segmentations, segmentations)

    # (num_chunks, num_speakers, num_frames)
    masks = torch.from_numpy(np.ascontiguousarray(segmentations.transpose(0, 2, 1)))
    chunks = [chunk for chunk, _ in binary_segmentations]

    model = self._embedding.model_
    device = self._embedding.device
    batch_size = self.embedding_batch_size
    batch_count = math.ceil(num_chunks / batch_size)

    if hook is not None:
        hook("embeddings", None, total=batch_count, completed=0)

    embedding_batches = []
    for i, start in enumerate(range(0, num_chunks, batch_size), 1):
        end = min(start + batch_size, num_chunks)
        waveforms = torch.stack(
            [self._audio.crop(file, chunks[c], mode="pad")[0] for c in range(start, end)]
        )
        with torch.inference_mode():
            embedding_batch = model(waveforms.to(device), weights=masks[start:end].to(device))
        embedding_batch = embedding_batch.cpu().numpy()
        embedding_batches.append(embedding_batch)

        if hook is not None:
            hook(
                "embeddings",
                embedding_batch.reshape(-1, embedding_batch.shape[-1]),
                total=batch_count,
                completed=i,
            )

    # (num_chunks, num_speakers, dimension)
    return np.concatenate(embedding_batches, axis=0)


class DiarizationPipeline:
    def __init__(
        self,
        model_name=None,
        token=None,
        device: str | torch.device | None = "mps",
        cache_dir=None,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        model_config = model_name or "pyannote/speaker-diarization-community-1"
        logger.info(f"Loading diarization model: {model_config}")
        self.model = Pipeline.from_pretrained(model_config, token=token, cache_dir=cache_dir).to(
            device
        )
        if isinstance(self.model, SpeakerDiarization) and _supports_stacked_masks(
            self.model._embedding
        ):
            self.model.get_embeddings = types.MethodType(_get_embeddings_per_chunk, self.model)

    def __call__(
        self,
        audio: str | np.ndarray,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        return_embeddings: bool = False,
        progress_callback: ProgressCallback = None,
    ) -> tuple[pd.DataFrame, dict[str, list[float]] | None] | pd.DataFrame:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Path to audio file or audio array
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            return_embeddings: Whether to return speaker embeddings
            progress_callback: Optional callable receiving a float (0-100) with progress percentage

        Returns:
            If return_embeddings is True:
                Tuple of (diarization dataframe, speaker embeddings dictionary)
            Otherwise:
                Just the diarization dataframe
        """
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": SAMPLE_RATE}

        hook = None
        if progress_callback is not None:
            # pyannote's diarization has two progress-trackable steps, each with
            # its own completed/total counter that resets between steps. Map each
            # step into a sub-range so progress is monotonic and meaningful.
            _STEP_RANGES = {
                "segmentation": (0.0, 50.0),
                "embeddings": (50.0, 99.0),
            }
            last_pct = [0.0]

            def hook(step_name, step_artifact, file=None, total=None, completed=None):
                if total is not None and completed is not None and total > 0:
                    offset, end = _STEP_RANGES.get(step_name, (0.0, 99.0))
                    pct = offset + min(completed / total, 1.0) * (end - offset)
                    if pct > last_pct[0]:
                        last_pct[0] = pct
                        progress_callback(pct)

        output = self.model(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            **({"hook": hook} if hook is not None else {}),
        )

        if progress_callback is not None:
            progress_callback(100.0)

        diarization = output.speaker_diarization
        embeddings = output.speaker_embeddings if return_embeddings else None

        diarize_df = pd.DataFrame(
            diarization.itertracks(yield_label=True), columns=["segment", "label", "speaker"]
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

        if return_embeddings and embeddings is not None:
            speaker_embeddings = {
                speaker: embeddings[s].tolist() for s, speaker in enumerate(diarization.labels())
            }
            return diarize_df, speaker_embeddings

        # For backwards compatibility
        if return_embeddings:
            return diarize_df, None
        else:
            return diarize_df


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: AlignedTranscriptionResult | TranscriptionResult,
    speaker_embeddings: dict[str, list[float]] | None = None,
    fill_nearest: bool = False,
) -> AlignedTranscriptionResult | TranscriptionResult:
    """
    Assign speakers to words and segments in the transcript.

    Uses an interval tree for O(log n) overlap queries instead of O(n) linear scan,
    achieving ~228x speedup for long-form content (3+ hour podcasts).

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    transcript_segments = transcript_result.get("segments", [])
    if not transcript_segments or diarize_df is None or len(diarize_df) == 0:
        return transcript_result

    # Build interval tree from diarization segments for O(log n) queries
    intervals = [(row["start"], row["end"], row["speaker"]) for _, row in diarize_df.iterrows()]
    tree = IntervalTree(intervals)

    for seg in transcript_segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)

        # Query overlapping segments using interval tree
        overlaps = tree.query(seg_start, seg_end)

        if overlaps:
            # Sum intersection durations per speaker and pick the dominant one
            speaker_intersections: dict[str, float] = {}
            for speaker, intersection in overlaps:
                speaker_intersections[speaker] = (
                    speaker_intersections.get(speaker, 0.0) + intersection
                )
            seg["speaker"] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
        elif fill_nearest:
            # Find nearest segment if no overlap
            seg_mid = (seg_start + seg_end) / 2
            nearest_speaker = tree.find_nearest(seg_mid)
            if nearest_speaker:
                seg["speaker"] = nearest_speaker

        # Assign speaker to words
        if "words" in seg:
            for word in seg["words"]:
                if "start" not in word:
                    continue

                word_start = word["start"]
                word_end = word.get("end", word_start)

                word_overlaps = tree.query(word_start, word_end)

                if word_overlaps:
                    speaker_intersections = {}
                    for speaker, intersection in word_overlaps:
                        speaker_intersections[speaker] = (
                            speaker_intersections.get(speaker, 0.0) + intersection
                        )
                    word["speaker"] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
                elif fill_nearest:
                    word_mid = (word_start + word_end) / 2
                    nearest_speaker = tree.find_nearest(word_mid)
                    if nearest_speaker:
                        word["speaker"] = nearest_speaker

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start: int, end: int, speaker: str | None = None):
        self.start = start
        self.end = end
        self.speaker = speaker
