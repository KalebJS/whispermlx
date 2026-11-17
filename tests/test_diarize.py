"""Tests for whispermlx/diarize.py — IntervalTree, assign_word_speakers, per-chunk embeddings."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from pyannote.audio.core.io import Audio
from pyannote.audio.models.embedding import WeSpeakerResNet34
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.speaker_verification import PyannoteAudioPretrainedSpeakerEmbedding
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature

from whispermlx.diarize import IntervalTree
from whispermlx.diarize import _get_embeddings_per_chunk
from whispermlx.diarize import _supports_stacked_masks
from whispermlx.diarize import assign_word_speakers


class TestIntervalTree:
    def test_empty_tree_query_returns_empty(self):
        tree = IntervalTree([])
        assert tree.query(0.0, 5.0) == []

    def test_empty_tree_find_nearest_returns_none(self):
        tree = IntervalTree([])
        assert tree.find_nearest(2.0) is None

    def test_empty_arrays_on_construction(self):
        tree = IntervalTree([])
        assert len(tree.starts) == 0
        assert len(tree.ends) == 0
        assert tree.speakers == []

    def test_single_interval_overlap(self):
        tree = IntervalTree([(0.0, 5.0, "A")])
        results = tree.query(2.0, 4.0)
        assert len(results) == 1
        speaker, duration = results[0]
        assert speaker == "A"
        assert duration == pytest.approx(2.0)

    def test_single_interval_no_overlap(self):
        tree = IntervalTree([(0.0, 5.0, "A")])
        assert tree.query(6.0, 8.0) == []

    def test_query_returns_empty_when_query_end_at_segment_start(self):
        """Open-interval semantics: query end == segment start → no overlap."""
        tree = IntervalTree([(5.0, 10.0, "A")])
        results = tree.query(0.0, 5.0)
        assert results == []

    def test_query_returns_empty_when_query_start_at_segment_end(self):
        """segment.start < query.end required for overlap."""
        tree = IntervalTree([(0.0, 5.0, "A")])
        results = tree.query(5.0, 10.0)
        assert results == []

    def test_two_intervals_both_overlap(self):
        tree = IntervalTree([(0.0, 6.0, "A"), (4.0, 10.0, "B")])
        results = tree.query(3.0, 7.0)
        speakers = {r[0] for r in results}
        assert "A" in speakers
        assert "B" in speakers

    def test_intersection_duration_correct(self):
        # A: [0, 6], B: [4, 10], query [3, 7]
        # A ∩ [3,7] = [3,6] → 3.0; B ∩ [3,7] = [4,7] → 3.0
        tree = IntervalTree([(0.0, 6.0, "A"), (4.0, 10.0, "B")])
        results = dict(tree.query(3.0, 7.0))
        assert results["A"] == pytest.approx(3.0)
        assert results["B"] == pytest.approx(3.0)

    def test_sorted_by_start_time(self):
        intervals = [(5.0, 8.0, "B"), (1.0, 3.0, "A"), (10.0, 12.0, "C")]
        tree = IntervalTree(intervals)
        starts = tree.starts.tolist()
        assert starts == sorted(starts)

    def test_find_nearest_uses_midpoints(self):
        # A midpoint=1.0, B midpoint=9.0; query at 7 → B is nearer
        tree = IntervalTree([(0.0, 2.0, "A"), (8.0, 10.0, "B")])
        assert tree.find_nearest(7.0) == "B"

    def test_find_nearest_single_interval(self):
        tree = IntervalTree([(2.0, 4.0, "A")])
        assert tree.find_nearest(100.0) == "A"

    def test_find_nearest_before_all_segments(self):
        tree = IntervalTree([(10.0, 20.0, "A"), (30.0, 40.0, "B")])
        # midpoints 15 and 35; time=0 → nearest is A
        assert tree.find_nearest(0.0) == "A"

    def test_large_dataset_correct_result(self):
        """1000 non-overlapping 1-second intervals; spot-check one query."""
        intervals = [(float(i), float(i + 1), f"S{i:04d}") for i in range(1000)]
        tree = IntervalTree(intervals)
        results = tree.query(500.0, 501.0)
        assert len(results) == 1
        assert results[0][0] == "S0500"

    def test_partial_overlap_duration(self):
        # segment [0, 10], query [8, 15] → intersection = 2.0
        tree = IntervalTree([(0.0, 10.0, "A")])
        results = tree.query(8.0, 15.0)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(2.0)


class TestAssignWordSpeakers:
    def test_empty_transcript_returns_unchanged(self, simple_diarize_df):
        result = {"segments": [], "language": "en"}
        out = assign_word_speakers(simple_diarize_df, result)
        assert out["segments"] == []

    def test_empty_diarize_df_returns_unchanged(self, minimal_transcript):
        empty_df = pd.DataFrame(columns=["start", "end", "speaker"])
        out = assign_word_speakers(empty_df, minimal_transcript)
        assert out is minimal_transcript

    def test_none_diarize_df_returns_unchanged(self, minimal_transcript):
        out = assign_word_speakers(None, minimal_transcript)
        assert out is minimal_transcript

    def test_segment_speaker_assigned(self, simple_diarize_df, minimal_transcript):
        # segment [1.0, 4.0] overlaps SPEAKER_00 [0.0, 5.0]
        result = copy.deepcopy(minimal_transcript)
        out = assign_word_speakers(simple_diarize_df, result)
        assert out["segments"][0].get("speaker") == "SPEAKER_00"

    def test_second_segment_speaker_assigned(self, simple_diarize_df, minimal_transcript):
        # segment [6.0, 9.0] overlaps SPEAKER_01 [5.0, 10.0]
        result = copy.deepcopy(minimal_transcript)
        out = assign_word_speakers(simple_diarize_df, result)
        assert out["segments"][1].get("speaker") == "SPEAKER_01"

    def test_word_speaker_assigned(self, simple_diarize_df, aligned_transcript):
        result = copy.deepcopy(aligned_transcript)
        out = assign_word_speakers(simple_diarize_df, result)
        # words at 1.0-2.5 and 2.5-4.0 are in SPEAKER_00 window
        assert out["segments"][0]["words"][0].get("speaker") == "SPEAKER_00"
        assert out["segments"][0]["words"][1].get("speaker") == "SPEAKER_00"

    def test_dominant_speaker_wins_with_overlap(self, overlapping_diarize_df):
        # SPEAKER_00: [0, 6], SPEAKER_01: [4, 10]
        # segment [3, 7] → A intersects [3,6]=3.0, B intersects [4,7]=3.0 — tie; max picks first alphabetically or first seen
        result = {
            "segments": [{"start": 5.0, "end": 9.0, "text": "overlap test"}],
            "language": "en",
        }
        out = assign_word_speakers(overlapping_diarize_df, result)
        # [5,9] ∩ SPEAKER_00[0,6] = 1.0, ∩ SPEAKER_01[4,10] = 4.0 → SPEAKER_01 wins
        assert out["segments"][0].get("speaker") == "SPEAKER_01"

    def test_fill_nearest_assigns_when_no_overlap(self, simple_diarize_df):
        # segment [11, 12] is outside both speakers (SPEAKER_00 ends at 5, SPEAKER_01 ends at 10)
        result = {
            "segments": [{"start": 11.0, "end": 12.0, "text": "beyond"}],
            "language": "en",
        }
        out = assign_word_speakers(simple_diarize_df, result, fill_nearest=True)
        assert "speaker" in out["segments"][0]

    def test_no_fill_nearest_leaves_unmatched_without_speaker(self, simple_diarize_df):
        result = {
            "segments": [{"start": 11.0, "end": 12.0, "text": "beyond"}],
            "language": "en",
        }
        out = assign_word_speakers(simple_diarize_df, result, fill_nearest=False)
        assert "speaker" not in out["segments"][0]

    def test_speaker_embeddings_attached(self, simple_diarize_df, minimal_transcript):
        embeddings = {"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]}
        result = copy.deepcopy(minimal_transcript)
        out = assign_word_speakers(simple_diarize_df, result, speaker_embeddings=embeddings)
        assert out["speaker_embeddings"] == embeddings

    def test_word_without_start_key_skipped(self, simple_diarize_df):
        result = {
            "segments": [
                {
                    "start": 1.0,
                    "end": 4.0,
                    "text": "hi",
                    "words": [
                        {"word": "hi", "score": 0.9},  # no start/end
                    ],
                }
            ],
            "language": "en",
        }
        # should not raise
        out = assign_word_speakers(simple_diarize_df, result)
        assert "speaker" not in out["segments"][0]["words"][0]

    def test_speaker_embeddings_none_not_attached(self, simple_diarize_df, minimal_transcript):
        result = copy.deepcopy(minimal_transcript)
        out = assign_word_speakers(simple_diarize_df, result, speaker_embeddings=None)
        assert "speaker_embeddings" not in out

    def test_returns_same_object(self, simple_diarize_df, minimal_transcript):
        result = copy.deepcopy(minimal_transcript)
        out = assign_word_speakers(simple_diarize_df, result)
        assert out is result


# ── Per-chunk embedding extraction ─────────────────────────────────────────────


class _FakeDiarization:
    """Just the attributes SpeakerDiarization.get_embeddings reads at inference time."""

    def __init__(self, embedding, batch_size):
        self._embedding = embedding
        self._audio = Audio(sample_rate=16000)
        self.embedding_batch_size = batch_size
        self.training = False


@pytest.fixture(scope="module")
def wespeaker_embedding():
    torch.manual_seed(0)
    return PyannoteAudioPretrainedSpeakerEmbedding(WeSpeakerResNet34().eval())


@pytest.fixture
def chunked_inputs():
    """Random 8s waveform cut into 2s chunks every 0.5s, with 3 local speakers per chunk."""
    rng = np.random.default_rng(0)
    file = {
        "waveform": torch.from_numpy(rng.standard_normal((1, 8 * 16000)).astype(np.float32)),
        "sample_rate": 16000,
    }
    num_chunks, num_frames, num_speakers = 13, 50, 3
    data = (rng.random((num_chunks, num_frames, num_speakers)) > 0.5).astype(np.float32)
    data[:, :, 2] = 0.0  # a fully silent speaker slot, as is common in practice
    data[0, :5, 0] = np.nan  # partial stitching may produce NaN
    segmentations = SlidingWindowFeature(data, SlidingWindow(start=0.0, duration=2.0, step=0.5))
    return file, segmentations


@pytest.mark.parametrize("exclude_overlap", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_per_chunk_embeddings_match_pyannote(
    wespeaker_embedding, chunked_inputs, exclude_overlap, batch_size
):
    file, segmentations = chunked_inputs
    pipeline = _FakeDiarization(wespeaker_embedding, batch_size)

    expected = SpeakerDiarization.get_embeddings(
        pipeline, file, segmentations, exclude_overlap=exclude_overlap
    )
    actual = _get_embeddings_per_chunk(
        pipeline, file, segmentations, exclude_overlap=exclude_overlap
    )

    assert actual.shape == expected.shape == (13, 3, wespeaker_embedding.dimension)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5, equal_nan=True)


def test_per_chunk_embeddings_report_progress(wespeaker_embedding, chunked_inputs):
    file, segmentations = chunked_inputs
    calls = []

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        calls.append(
            (step_name, total, completed, None if step_artifact is None else step_artifact.shape)
        )

    _get_embeddings_per_chunk(
        _FakeDiarization(wespeaker_embedding, 4), file, segmentations, hook=hook
    )

    dim = wespeaker_embedding.dimension
    assert calls == [
        ("embeddings", 4, 0, None),
        ("embeddings", 4, 1, (12, dim)),
        ("embeddings", 4, 2, (12, dim)),
        ("embeddings", 4, 3, (12, dim)),
        ("embeddings", 4, 4, (3, dim)),
    ]


def test_supports_stacked_masks_for_wespeaker(wespeaker_embedding):
    assert _supports_stacked_masks(wespeaker_embedding) is True


def test_supports_stacked_masks_rejects_other_backends():
    class OtherBackend:
        def __call__(self, waveforms, masks=None):
            return np.zeros((waveforms.shape[0], 192))

    assert _supports_stacked_masks(OtherBackend()) is False
