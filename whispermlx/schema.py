from collections.abc import Callable
from typing import TypedDict

ProgressCallback = Callable[[float], None] | None

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


class SingleWordSegment(TypedDict):
    """
    A single word of a speech.
    """

    word: str
    start: float
    end: float
    score: float


class SingleCharSegment(TypedDict):
    """
    A single char of a speech.
    """

    char: str
    start: float
    end: float
    score: float


class SingleSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech.
    """

    start: float
    end: float
    text: str
    avg_logprob: NotRequired[float]


class SegmentData(TypedDict):
    """
    Temporary processing data used during alignment.
    Contains cleaned and preprocessed data for each segment.
    """

    clean_char: list[str]  # Cleaned characters that exist in model dictionary
    clean_cdx: list[int]  # Original indices of cleaned characters
    clean_wdx: list[int]  # Indices of words containing valid characters
    sentence_spans: list[tuple[int, int]]  # Start and end indices of sentences


class SingleAlignedSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech with word alignment.
    """

    start: float
    end: float
    text: str
    avg_logprob: NotRequired[float]
    words: list[SingleWordSegment]
    chars: list[SingleCharSegment] | None


class TranscriptionResult(TypedDict):
    """
    A list of segments and word segments of a speech.
    """

    segments: list[SingleSegment]
    language: str


class AlignedTranscriptionResult(TypedDict):
    """
    A list of segments and word segments of a speech.
    """

    segments: list[SingleAlignedSegment]
    word_segments: list[SingleWordSegment]
