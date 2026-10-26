"""Extracts visual keyframe events from spoken deictic references in video streams."""

import logging
import uuid
from typing import Callable

import cv2
import imagehash
from PIL import Image
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from visualrag.deictic.dedup import DeicticDeduplicator
from visualrag.deictic.detector import DeicticDetector
from visualrag.deictic.synchronizer import DeicticKeyframeSelector
from visualrag.io import VideoInput, resolve_video_input, rewind_stream
from visualrag.models import DeicticEvent, DeicticIntent, TranscriptSegment
from visualrag.storage import CallbackStorage, MemoryStorage, StorageBackend

logger = logging.getLogger(__name__)


class DeicticExtractor:
    """Extracts visual keyframe events from spoken deictic references in videos.

    Scans transcript segments for spoken pointing references ("look at this", "plug here", "click here"),
    samples candidate video frames near the speech gesture apex using kinematic hold detection,
    deduplicates redundant references using perceptual hashing, and registers keyframe image URIs.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        validate_models: bool = True,
        storage_backend: StorageBackend | Callable[[bytes, str], str | None] | None = None,
        dedup_window_sec: float = 45.0,
        max_hash_distance: int = 5,
        burst_frame_count: int = 5,
    ) -> None:
        """Initialize the deictic event extraction pipeline.

        Args:
            llm: Optional chat model with structured output support for intent parsing.
            validate_models: Whether to verify LLM capability upfront.
            storage_backend: Storage mechanism for saving extracted keyframe images.
            dedup_window_sec: Sliding window duration in seconds for collapsing repeated references.
            max_hash_distance: Maximum Hamming distance for perceptual image deduplication.
            burst_frame_count: Number of candidate frames to sample around each gesture apex.
        """
        self.llm = llm

        # Fail-fast capability validation
        if validate_models:
            self._validate_model_capabilities(llm)

        self.detector = DeicticDetector(llm=llm)
        self.synchronizer = DeicticKeyframeSelector(burst_frame_count=burst_frame_count)
        self.deduplicator = DeicticDeduplicator(
            dedup_window_sec=dedup_window_sec,
            max_hash_distance=max_hash_distance,
        )

        if storage_backend is not None:
            if callable(storage_backend) and not hasattr(storage_backend, "save"):
                self.storage: StorageBackend = CallbackStorage(storage_backend)
            else:
                self.storage = storage_backend
        else:
            self.storage = MemoryStorage()

    def _validate_model_capabilities(
        self,
        llm: BaseChatModel | None,
    ) -> None:
        """Validate LLM capability upfront to fail fast before video decoding."""
        if llm is not None:
            if not isinstance(llm, (BaseChatModel, Runnable)):
                raise ValueError(f"LLM must be a LangChain BaseChatModel or Runnable, got {type(llm)}")
            if hasattr(llm, "with_structured_output"):
                try:
                    _ = llm.with_structured_output(DeicticIntent)
                except Exception as exc:
                    logger.debug("Model does not natively bind with_structured_output: %s", exc)

    def extract(
        self,
        video_path: VideoInput,
        transcript_segments: list[TranscriptSegment],
        video_name: str | None = None,
    ) -> list[DeicticEvent]:
        """Extract all verified deictic events with keyframe visual grounding.

        Args:
            video_path: Path to the natural video file, seekable stream, or upload object.
            transcript_segments: List of timestamped transcript segments.
            video_name: Optional explicit name for the video.

        Returns:
            List of DeicticEvent objects in chronological order.
        """
        target, source_name, video_stem = resolve_video_input(video_path, video_name=video_name)
        self.deduplicator.reset()
        events: list[DeicticEvent] = []

        try:
            for segment in transcript_segments:
                # Deictic intent gate
                intent = self.detector.detect(segment)
                if not intent.is_visual_deictic:
                    continue

                target_phrase = intent.target_phrase or "referenced item"
                grounding_question = intent.grounding_question or f"What is shown regarding {target_phrase}?"

                # Kinematic keyframe selection (gesture hold apex)
                try:
                    keyframe_bgr, keyframe_pts = self.synchronizer.select_keyframe(
                        target,
                        segment,
                        target_phrase,
                        video_name=source_name,
                    )
                except Exception as exc:
                    logger.warning(
                        "Keyframe extraction failed at %.1fs: %s",
                        segment.start_sec,
                        exc,
                    )
                    continue

                # Lossless PNG encoding and perceptual hash
                success, buf = cv2.imencode(".png", keyframe_bgr)
                if not success:
                    logger.warning("Failed to encode keyframe to PNG at %.1fs", keyframe_pts)
                    continue
                image_bytes = buf.tobytes()

                pil_img = Image.fromarray(cv2.cvtColor(keyframe_bgr, cv2.COLOR_BGR2RGB))
                phash = str(imagehash.phash(pil_img))

                # Multi-level deduplication check
                is_dup, cached_event = self.deduplicator.check(
                    segment.start_sec,
                    phash,
                    target_phrase,
                    source_video=source_name,
                )

                if is_dup and cached_event is not None:
                    visual_text = cached_event.visual_grounding_text or cached_event.target_phrase
                    is_coref = True
                else:
                    visual_text = target_phrase
                    is_coref = False

                # Persist keyframe via storage backend
                filename = f"{video_stem}_deictic_{segment.start_sec:.1f}s_{uuid.uuid4().hex[:8]}.png"
                uri = self.storage.save(image_bytes, filename)

                event = DeicticEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp_sec=segment.start_sec,
                    trigger_utterance=segment.text,
                    target_phrase=target_phrase,
                    grounding_question=grounding_question,
                    keyframe_timestamp_sec=keyframe_pts,
                    image_bytes=image_bytes,
                    image_hash=phash,
                    visual_grounding_text=visual_text,
                    is_coreference_reuse=is_coref,
                    image_uri=uri,
                    metadata={
                        "source_video": source_name,
                        "start_time_sec": segment.start_sec,
                        "end_time_sec": segment.end_sec,
                    },
                )

                self.deduplicator.register(event)
                events.append(event)
        finally:
            rewind_stream(target)

        return events
