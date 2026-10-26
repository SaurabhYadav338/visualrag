"""Kinematic keyframe selection using Kendon's gesture hold (motion-minimum + sharpness)."""

import logging
import re
from typing import Any

import av
import cv2
import numpy as np

from visualrag.io import (
    VideoInput,
    open_video_container,
    resolve_video_input,
    rewind_stream,
)
from visualrag.models import TranscriptSegment

logger = logging.getLogger(__name__)


class DeicticKeyframeSelector:
    """Selects stable, low-blur video keyframes corresponding to spoken deictic references.

    Human pointing gestures typically pause momentarily at their apex ~200ms after
    the spoken keyword. Candidate frames across this window are scored by maximizing
    sharpness (Laplacian variance) while minimizing inter-frame motion.
    """

    _DEICTIC_KEYWORDS = {
        # Core demonstratives & locatives
        "this", "these", "that", "those", "here", "there",
        # Attention triggers
        "look", "notice", "see", "watch", "check", "view",
        # Software & UI actions
        "click", "press", "tap", "touch", "hover", "drag", "drop",
        "select", "toggle", "highlight", "scroll", "swipe", "type",
        # Physical & general manipulation actions
        "plug", "connect", "attach", "insert", "turn", "push", "mount", "place",
    }

    def __init__(
        self,
        burst_frame_count: int = 5,
        gesture_hold_offset_sec: float = 0.2,
        window_pre_sec: float = 0.2,
        window_post_sec: float = 0.4,
        epsilon: float = 1e-4,
    ) -> None:
        """Initialize the keyframe synchronizer.

        Args:
            burst_frame_count: Number of candidate frames to decode across the temporal search window.
            gesture_hold_offset_sec: Expected lag in seconds between spoken keyword and physical gesture apex.
            window_pre_sec: Search window duration in seconds before the estimated gesture apex.
            window_post_sec: Search window duration in seconds after the estimated gesture apex.
            epsilon: Small constant to prevent division by zero in motion score normalization.
        """
        self.burst_frame_count = burst_frame_count
        self.gesture_hold_offset_sec = gesture_hold_offset_sec
        self.window_pre_sec = window_pre_sec
        self.window_post_sec = window_post_sec
        self.epsilon = epsilon

    def find_target_timestamp(
        self,
        segment: TranscriptSegment,
        target_phrase: str | None = None,
    ) -> float:
        """Find the anchor timestamp t_target from word timestamps or segment timing."""
        if segment.words:
            # Match timestamp of target phrase or deictic trigger words
            phrase_words = set(re.findall(r"\w+", target_phrase.lower())) if target_phrase else set()
            for word_obj in segment.words:
                clean_word = re.sub(r"[^\w]", "", word_obj.word.lower())
                if clean_word in phrase_words or clean_word in self._DEICTIC_KEYWORDS:
                    return word_obj.start_sec + self.gesture_hold_offset_sec

            return segment.words[0].start_sec + self.gesture_hold_offset_sec

        return (segment.start_sec + segment.end_sec) / 2.0

    def score_frames(
        self,
        frames: list[np.ndarray],
    ) -> list[float]:
        """Compute the Kendon hold score for each frame in the candidate burst."""
        if not frames:
            return []

        grays = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
            for f in frames
        ]

        # Laplacian variance measures edge sharpness
        lap_vars = [
            float(cv2.Laplacian(g, cv2.CV_64F).var())
            for g in grays
        ]

        # Inter-frame difference estimates camera/gesture motion velocity
        motion_diffs: list[float] = []
        for i, g in enumerate(grays):
            if i == 0:
                if len(grays) > 1:
                    diff = float(np.mean(cv2.absdiff(grays[0], grays[1])) / 255.0)
                else:
                    diff = 0.0
            else:
                diff = float(np.mean(cv2.absdiff(grays[i], grays[i - 1])) / 255.0)
            motion_diffs.append(diff)

        # Kendon score: Sharpness / (MotionDiff + epsilon)
        scores = [
            lap_vars[i] / (motion_diffs[i] + self.epsilon)
            for i in range(len(frames))
        ]
        return scores

    def select_from_frames(
        self,
        frames_with_times: list[tuple[np.ndarray, float]],
    ) -> tuple[np.ndarray, float]:
        """Select best keyframe from a list of (frame_bgr, timestamp_sec) pairs."""
        if not frames_with_times:
            raise ValueError("frames_with_times list cannot be empty")

        if len(frames_with_times) == 1:
            return frames_with_times[0]

        frames = [f for f, _ in frames_with_times]
        scores = self.score_frames(frames)
        best_idx = int(np.argmax(scores))
        return frames_with_times[best_idx]

    def select_keyframe(
        self,
        video_path: VideoInput,
        segment: TranscriptSegment,
        target_phrase: str | None = None,
        video_name: str | None = None,
    ) -> tuple[np.ndarray, float]:
        """Extract candidate burst from video and select the Kendon hold apex frame."""
        t_target = self.find_target_timestamp(segment, target_phrase)
        t_start = max(0.0, t_target - self.window_pre_sec)
        t_end = t_target + self.window_post_sec

        target, source_name, _ = resolve_video_input(video_path, video_name=video_name)
        burst = self._decode_burst(target, t_start, t_end, self.burst_frame_count)
        if not burst:
            raise RuntimeError(f"Could not decode video burst from {source_name} between {t_start:.2f}s and {t_end:.2f}s")

        return self.select_from_frames(burst)

    def _decode_burst(
        self,
        target: Any,
        t_start: float,
        t_end: float,
        num_frames: int,
    ) -> list[tuple[np.ndarray, float]]:
        """Decode num_frames evenly distributed between t_start and t_end."""
        target_times = np.linspace(t_start, t_end, num_frames).tolist()
        collected: list[tuple[np.ndarray, float]] = []

        container = open_video_container(target)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            # Seek close to t_start to avoid decoding from frame 0
            try:
                target_pts = int(max(0.0, t_start - 0.5) / stream.time_base)
                container.seek(target_pts, stream=stream)
            except Exception as seek_err:
                logger.debug("Seek before burst decode failed: %s", seek_err)

            target_idx = 0
            for frame in container.decode(video=0):
                pts_sec = float(frame.pts * stream.time_base)
                if pts_sec >= target_times[target_idx]:
                    bgr = frame.to_ndarray(format="bgr24")
                    collected.append((bgr, pts_sec))
                    target_idx += 1
                    if target_idx >= len(target_times):
                        break
        finally:
            container.close()
            rewind_stream(target)

        return collected
