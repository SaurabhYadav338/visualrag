"""VisualRAG - Unified top-level entry point for multimodal video RAG."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from visualrag.classifier import VideoModalityClassifier
from visualrag.deictic.extractor import DeicticExtractor
from visualrag.io import VideoInput, resolve_video_input
from visualrag.models import (
    DeicticEvent,
    ScreenFrame,
    TranscriptSegment,
    VideoModality,
    VisualSegment,
)
from visualrag.screen.aligner import align_transcript
from visualrag.screen.extractor import ScreenExtractor
from visualrag.screen.ocr import enrich_frames_with_ocr
from visualrag.storage import CallbackStorage, LocalStorage, MemoryStorage, StorageBackend

logger = logging.getLogger(__name__)


class VisualRAG:
    """Processes meeting presentations and video walkthroughs into structured visual segments.

    Auto-detects whether a video is a screen recording or physical camera
    footage, extracting slides with OCR for presentations and synchronized keyframes
    for spoken physical demonstrations.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        storage_backend: StorageBackend | None = None,
        output_dir: str | Path | None = None,
        slide_sample_interval: float = 1.0,
        slide_change_threshold: float = 0.10,
        dedup_window_sec: float = 45.0,
        max_hash_distance: int = 5,
        classifier: VideoModalityClassifier | None = None,
    ) -> None:
        """Initialize the unified VisualRAG pipeline.

        Args:
            llm: Optional chat model for extracting deictic intents and visual questions.
            storage_backend: Custom storage backend for saving extracted images.
            output_dir: Local filesystem directory for saving images if no backend provided.
            slide_sample_interval: PTS interval in seconds between frame decodes for screen recordings.
            slide_change_threshold: Normalized pixel change threshold triggering a slide cut.
            dedup_window_sec: Temporal window in seconds for deduplicating repeated deictic events.
            max_hash_distance: Maximum perceptual hash Hamming distance for visual deduplication.
            classifier: Optional classifier instance to override video modality detection heuristics.
        """
        self.llm = llm

        if storage_backend is not None:
            if callable(storage_backend) and not hasattr(storage_backend, "save"):
                self.storage: StorageBackend = CallbackStorage(storage_backend)
            else:
                self.storage = storage_backend
        elif output_dir is not None:
            self.storage = LocalStorage(output_dir=Path(output_dir))
        else:
            self.storage = MemoryStorage()

        self.classifier = classifier or VideoModalityClassifier()
        self.screen_extractor = ScreenExtractor(
            sample_interval=slide_sample_interval,
            storage_backend=self.storage,
            change_threshold=slide_change_threshold,
        )
        self.deictic_extractor = DeicticExtractor(
            llm=self.llm,
            validate_models=False,
            storage_backend=self.storage,
            dedup_window_sec=dedup_window_sec,
            max_hash_distance=max_hash_distance,
        )

    def classify(
        self,
        video: VideoInput,
        video_name: str | None = None,
    ) -> VideoModality:
        """Auto-detect video modality (SCREEN_RECORDING vs. NATURAL_VIDEO)."""
        return self.classifier.classify(video, video_name=video_name)

    def process(
        self,
        video: VideoInput,
        transcript: list[TranscriptSegment] | None = None,
        modality: VideoModality | None = None,
        video_name: str | None = None,
    ) -> list[VisualSegment]:
        """Process an input video into unified VisualSegments.

        Automatically routes between slide extraction and deictic visual grounding
        based on the detected video modality. Accepts file paths, binary streams,
        and web upload objects.

        Args:
            video: Path to video file, seekable binary stream (BinaryIO/BytesIO), or upload object.
            transcript: Optional timestamped transcript segments (required for natural videos).
            modality: Explicit modality override. If None, auto-detected via classifier.
            video_name: Optional explicit name for the video.

        Returns:
            List of VisualSegment objects carrying visual content, timestamps, and image URIs.
        """
        target, source_name, video_stem = resolve_video_input(video, video_name=video_name)
        if isinstance(target, str) and not Path(target).exists():
            raise FileNotFoundError(f"Video file not found at: {target}")

        resolved_modality = modality or self.classify(target, video_name=source_name)
        logger.info("Processing %s with modality %s", source_name, resolved_modality.value)

        if resolved_modality == VideoModality.SCREEN_RECORDING:
            screens = self.screen_extractor.extract(target, video_name=source_name)
            enrich_frames_with_ocr(screens)
            if transcript:
                align_transcript(transcript, screens)
            return [
                VisualSegment(
                    id=s.frame_id,
                    modality=VideoModality.SCREEN_RECORDING,
                    source_video=s.source_video,
                    start_time_sec=s.start_time_sec,
                    end_time_sec=s.end_time_sec if s.end_time_sec is not None else s.start_time_sec,
                    keyframe_timestamp_sec=s.start_time_sec,
                    image_uri=s.image_uri,
                    image_hash=s.image_hash,
                    text=s.ocr_text,
                    transcript=s.transcript_segment,
                    metadata={
                        "frame_index": s.frame_index,
                        "ocr_confidence": s.ocr_confidence,
                        "bbox": s.bbox,
                        **s.metadata,
                    },
                )
                for s in screens
            ]

        else:
            if not transcript:
                logger.warning(
                    "Processing natural video '%s' without a transcript. "
                    "Deictic events require spoken segments for audio-visual grounding.",
                    source_name,
                )
            events = self.deictic_extractor.extract(target, transcript or [], video_name=source_name)
            return [
                VisualSegment(
                    id=e.event_id,
                    modality=VideoModality.NATURAL_VIDEO,
                    source_video=str(e.metadata.get("source_video", source_name)),
                    start_time_sec=e.timestamp_sec,
                    end_time_sec=float(e.metadata.get("end_time_sec", e.timestamp_sec)),
                    keyframe_timestamp_sec=e.keyframe_timestamp_sec,
                    image_uri=e.image_uri,
                    image_hash=e.image_hash,
                    text=e.visual_grounding_text,
                    transcript=e.trigger_utterance,
                    metadata={
                        "target_phrase": e.target_phrase,
                        "grounding_question": e.grounding_question,
                        "is_coreference_reuse": e.is_coreference_reuse,
                        **e.metadata,
                    },
                )
                for e in events
            ]

    def extract_screen(
        self,
        video: VideoInput,
        ocr: bool = True,
        transcript: list[TranscriptSegment] | None = None,
        video_name: str | None = None,
    ) -> list[ScreenFrame]:
        """Extract screen frames from a screen recording video, stream, or upload object."""
        screens = self.screen_extractor.extract(video, video_name=video_name)
        if ocr:
            enrich_frames_with_ocr(screens)
        if transcript:
            align_transcript(transcript, screens)
        return screens

    def extract_deictic_events(
        self,
        video: VideoInput,
        transcript: list[TranscriptSegment],
        video_name: str | None = None,
    ) -> list[DeicticEvent]:
        """Extract grounded deictic keyframe events from a natural video, stream or upload."""
        return self.deictic_extractor.extract(video, transcript, video_name=video_name)
