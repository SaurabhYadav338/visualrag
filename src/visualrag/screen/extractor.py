"""Extracts individual slide transitions and crops from video streams."""

import logging
import uuid
from pathlib import Path
from typing import Callable

import av
import cv2
import imagehash
import numpy as np
from PIL import Image

from visualrag.io import (
    VideoInput,
    open_video_container,
    resolve_video_input,
    rewind_stream,
)
from visualrag.models import ScreenFrame
from visualrag.screen.cropper import DynamicCropper
from visualrag.storage import CallbackStorage, LocalStorage, MemoryStorage, StorageBackend

logger = logging.getLogger(__name__)


class ScreenExtractor:
    """Decodes video streams and extracts unique presentation slides and screenshare states.

    Samples frames using Presentation Timestamps (PTS) rather than fixed
    frame counts to handle Variable Frame Rate (VFR) meeting recordings.
    Consecutive cropped slide regions are compared using blurred absolute
    pixel differences and perceptual hashing to identify true slide changes
    while ignoring presenter webcams, animations, and transient mouse motion.
    """

    def __init__(
        self,
        sample_interval: float = 1.0,
        change_threshold: float = 0.10,
        cooldown_sec: float = 2.0,
        dedup_interval_sec: float = 120.0,
        output_dir: Path | str | None = "slides",
        min_area_ratio: float = 0.25,
        max_hash_distance: int = 2,
        storage_backend: StorageBackend | Callable[[bytes, str], str | None] | None = None,
    ) -> None:
        """Initialize the screen extractor.

        Args:
            sample_interval: Minimum presentation timestamp (PTS) gap between sampled frames in seconds.
            change_threshold: Fraction of pixels differing (after blur) to declare a slide change.
            cooldown_sec: Minimum duration in seconds to suppress new cuts following a slide change.
            dedup_interval_sec: Window in seconds within which duplicate slide hashes are suppressed.
            output_dir: Directory where slide images are saved if using local filesystem storage.
            min_area_ratio: Minimum screen fraction a detected region must occupy to qualify as a slide.
            max_hash_distance: Maximum perceptual hash Hamming distance for near-duplicate rejection.
            storage_backend: Optional backend or callable callback for saving slide image buffers.
        """
        self._sample_interval = sample_interval
        self._change_threshold = change_threshold
        self._cooldown_sec = cooldown_sec
        self._dedup_interval_sec = dedup_interval_sec
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._max_hash_distance = max_hash_distance
        self._cropper = DynamicCropper(min_area_ratio=min_area_ratio)

        if storage_backend is not None:
            if callable(storage_backend) and not hasattr(storage_backend, "save"):
                self._storage: StorageBackend = CallbackStorage(storage_backend)
            else:
                self._storage = storage_backend
        elif self._output_dir is not None:
            self._storage = LocalStorage(self._output_dir)
        else:
            self._storage = MemoryStorage()

    def extract(
        self,
        video_path: VideoInput,
        video_name: str | None = None,
    ) -> list[ScreenFrame]:
        """Extract all unique screenshare frame/presentation slides from a video file, stream, or upload.

        Args:
            video_path: File path, open binary stream, or framework upload object.
            video_name: Optional explicit display/reference name for the video.

        Returns:
            List of ScreenFrame objects in chronological order.
        """
        target, source_name, video_stem = resolve_video_input(video_path, video_name=video_name)
        if isinstance(self._storage, LocalStorage):
            self._storage.output_dir.mkdir(parents=True, exist_ok=True)
        self._cropper.reset()

        slides: list[ScreenFrame] = []
        prev_crop_gray: np.ndarray | None = None
        last_transition_sec: float = -self._cooldown_sec
        last_pts_sec: float = -float("inf")
        hash_history: list[tuple[imagehash.ImageHash, float]] = []

        try:
            container = open_video_container(target)
        except Exception as exc:
            logger.error("Failed to open video %s: %s", source_name, exc)
            return slides

        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            for frame in container.decode(video=0):
                try:
                    current_sec = float(frame.pts * stream.time_base)

                    # Timestamp-based sampling
                    if current_sec - last_pts_sec < self._sample_interval:
                        continue
                    last_pts_sec = current_sec

                    bgr = frame.to_ndarray(format="bgr24")
                    bbox = self._cropper.process_frame(bgr)
                    if bbox is None:
                        # Presentation is inactive; pause visual diff tracking
                        prev_crop_gray = None
                        continue

                    x, y, w, h = bbox
                    crop_bgr = bgr[y : y + h, x : x + w]

                    # Grayscale and blur to ignore minor cursor motion and compression noise
                    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                    crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 0)

                    if prev_crop_gray is None:
                        # First visible slide after start or gap
                        prev_crop_gray = crop_gray
                        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                        phash = imagehash.phash(pil_img)
                        slide = self._save_slide(
                            crop_bgr,
                            current_sec,
                            video_stem,
                            source_name,
                            slides,
                            bbox,
                            image_hash=str(phash),
                        )
                        if slide is not None:
                            slides.append(slide)
                            hash_history.append((phash, current_sec))
                            last_transition_sec = current_sec
                        continue

                    # Handle layout or aspect ratio changes mid-video
                    if prev_crop_gray.shape != crop_gray.shape:
                        prev_crop_gray = cv2.resize(
                            prev_crop_gray,
                            (crop_gray.shape[1], crop_gray.shape[0]),
                        )

                    diff = cv2.absdiff(prev_crop_gray, crop_gray)
                    changed_fraction = np.count_nonzero(diff > 25) / diff.size

                    if changed_fraction > self._change_threshold:
                        # Enforce cooldown to ignore animation frames
                        if current_sec - last_transition_sec < self._cooldown_sec:
                            prev_crop_gray = crop_gray
                            continue

                        # Perceptual hash comparison to avoid redundant slides
                        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                        phash = imagehash.phash(pil_img)

                        is_duplicate = False
                        for seen_hash, last_seen_sec in hash_history:
                            if phash - seen_hash <= self._max_hash_distance:
                                if current_sec - last_seen_sec < self._dedup_interval_sec:
                                    is_duplicate = True
                                    logger.debug(
                                        "Duplicate slide skipped at %.1fs (distance %d <= %d)",
                                        current_sec,
                                        phash - seen_hash,
                                        self._max_hash_distance,
                                    )
                                    break

                        if is_duplicate:
                            prev_crop_gray = crop_gray
                            continue

                        # Backfill duration for previous slide
                        if slides:
                            slides[-1].end_time_sec = current_sec

                        slide = self._save_slide(
                            crop_bgr,
                            current_sec,
                            video_stem,
                            source_name,
                            slides,
                            bbox,
                            image_hash=str(phash),
                        )
                        if slide is not None:
                            slides.append(slide)
                            hash_history.append((phash, current_sec))
                            last_transition_sec = current_sec

                    prev_crop_gray = crop_gray

                except Exception as exc:
                    logger.warning("Skipping frame near %.1fs: %s", last_pts_sec, exc)
                    continue

            # When EOF is reached, ensure the last slide has an accurate end_time_sec
            if slides and (slides[-1].end_time_sec is None or slides[-1].end_time_sec < last_pts_sec):
                total_duration_sec = None
                if stream.duration and stream.time_base:
                    total_duration_sec = float(stream.duration * stream.time_base)
                elif container.duration:
                    total_duration_sec = float(container.duration / av.time_base)

                final_sec = (
                    total_duration_sec
                    if (total_duration_sec is not None and total_duration_sec > last_pts_sec)
                    else max(last_pts_sec, slides[-1].start_time_sec)
                )
                slides[-1].end_time_sec = float(final_sec)

        except Exception as exc:
            logger.error("Error decoding video %s: %s", source_name, exc)
        finally:
            container.close()
            rewind_stream(target)

        return slides

    def _save_slide(
        self,
        crop_bgr: np.ndarray,
        current_sec: float,
        video_stem: str,
        source_video: str,
        existing_slides: list[ScreenFrame],
        bbox: tuple[int, int, int, int],
        image_hash: str = "",
    ) -> ScreenFrame | None:
        try:
            frame_index = len(existing_slides)
            frame_id = str(uuid.uuid4())

            success, buffer = cv2.imencode(".png", crop_bgr)
            if not success:
                logger.warning("Failed to encode slide image to PNG at %.1fs", current_sec)
                return None
            image_bytes = buffer.tobytes()

            if not image_hash:
                pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                image_hash = str(imagehash.phash(pil_img))

            filename = f"{video_stem}_frame{frame_index:04d}_{frame_id[:8]}.png"
            uri = self._storage.save(image_bytes, filename)
            return ScreenFrame(
                frame_id=frame_id,
                frame_index=frame_index,
                source_video=source_video,
                start_time_sec=current_sec,
                bbox=bbox,
                image_bytes=image_bytes,
                image_uri=uri,
                image_hash=image_hash,
            )
        except Exception as exc:
            logger.warning("Failed to save slide at %.1fs: %s", current_sec, exc)
            return None
