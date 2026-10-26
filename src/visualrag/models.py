"""Core data contracts and schemas."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VideoModality(str, Enum):
    """Categorization of video content structure."""

    SCREEN_RECORDING = "screen_recording"
    NATURAL_VIDEO = "natural_video"
    UNKNOWN = "unknown"


@dataclass
class TranscriptWord:
    """Individual word with sub-second alignment."""

    word: str
    start_sec: float
    end_sec: float
    probability: float = 1.0


@dataclass
class TranscriptSegment:
    """Utterance segment spanning a time interval."""

    start_sec: float
    end_sec: float
    text: str
    speaker: str | None = None
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass
class ScreenFrame:
    """Extracted screen transition containing OCR content, visual bounding box, and timestamps."""

    frame_id: str
    frame_index: int
    source_video: str

    start_time_sec: float
    end_time_sec: float | None = None

    image_bytes: bytes = b""
    image_uri: str | None = None
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    transcript_segment: str | None = None

    # Slide geometry in source video (x, y, width, height)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    # Perceptual hash & metadata
    image_hash: str = ""
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float | None:
        """Display duration in seconds if end_time_sec has been resolved."""
        if self.end_time_sec is not None:
            return self.end_time_sec - self.start_time_sec
        return None


class DeicticIntent(BaseModel):
    """Classification of visual deictic intent in spoken text."""

    is_visual_deictic: bool = Field(
        description="Whether the speaker refers to a physical object or UI element visually visible."
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    target_phrase: str | None = Field(
        default=None,
        description="Noun phrase or object referenced by the speaker.",
    )
    grounding_question: str | None = Field(
        default=None,
        description="Question directed to a vision model to resolve what is shown.",
    )


@dataclass
class DeicticEvent:
    """Extracted gesture event paired with its visual keyframe and description."""

    event_id: str
    timestamp_sec: float
    trigger_utterance: str
    target_phrase: str
    grounding_question: str
    keyframe_timestamp_sec: float

    # Keyframe data
    image_bytes: bytes = b""
    image_hash: str = ""

    # Visual grounding
    visual_grounding_text: str = ""
    is_coreference_reuse: bool = False

    # Optional storage URI
    image_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualSegment:
    """Unified visual extraction unit across both screen recordings and natural videos.

    Represents a temporally-bounded visual state (a presentation slide/screen
    or a physical deictic demonstration keyframe) with its associated text,
    transcript, and storage image URI.
    """

    # Identity & Modality
    id: str
    modality: VideoModality
    source_video: str

    # Temporal boundaries (guaranteed non-None timestamps in seconds)
    start_time_sec: float
    end_time_sec: float
    keyframe_timestamp_sec: float = 0.0

    # Visual Storage Reference
    image_uri: str | None = None
    image_hash: str = ""

    # Content
    text: str = ""  # OCR text for screen recordings, visual grounding description for natural video
    transcript: str | None = None  # Aligned spoken dialogue during this segment

    # Rich modality-specific metadata (bbox, ocr_confidence, target_phrase, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        """Display duration in seconds."""
        return max(0.0, self.end_time_sec - self.start_time_sec)

    def to_dict(self) -> dict[str, Any]:
        """Serialize segment to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "modality": self.modality.value,
            "source_video": self.source_video,
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "duration_sec": self.duration_sec,
            "keyframe_timestamp_sec": self.keyframe_timestamp_sec,
            "image_uri": self.image_uri,
            "image_hash": self.image_hash,
            "text": self.text,
            "transcript": self.transcript,
            "metadata": self.metadata,
        }
