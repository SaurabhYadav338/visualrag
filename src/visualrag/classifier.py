import logging
from pathlib import Path

import av
import cv2
import numpy as np

from visualrag.io import (
    VideoInput,
    open_video_container,
    resolve_video_input,
    rewind_stream,
)
from visualrag.models import VideoModality

logger = logging.getLogger(__name__)


class VideoModalityClassifier:
    """Classifies video input as a screen recording or physical camera footage.

    Samples frames across the video and evaluates structural heuristics:
    orthogonal edge proportion (high in window borders and slides), color flatness
    (solid UI backgrounds vs. continuous camera gradients), and typographic contour density.
    """

    def __init__(
        self,
        sample_frame_count: int = 5,
        orthogonal_threshold: float = 0.65,
        flat_color_threshold: float = 0.35,
        text_density_threshold: float = 3.0,
    ) -> None:
        """
        Args:
            sample_frame_count: Number of equidistant frames sampled across the video timeline.
            orthogonal_threshold: Minimum fraction of horizontal/vertical edges indicative of UI/slides.
            flat_color_threshold: Minimum proportion of flat color patches typical of digital slides.
            text_density_threshold: Minimum density of text-like contours per 10k pixels.
        """
        self.sample_frame_count = max(1, sample_frame_count)
        self.orthogonal_threshold = orthogonal_threshold
        self.flat_color_threshold = flat_color_threshold
        self.text_density_threshold = text_density_threshold

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict[str, float]:
        """Compute visual modality metrics for a single frame.

        Args:
            frame_bgr: BGR image frame as a NumPy array.

        Returns:
            Dictionary containing 'orthogonal_edge_ratio', 'flat_color_ratio',
            and 'text_contour_density'.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "orthogonal_edge_ratio": 0.0,
                "flat_color_ratio": 0.0,
                "text_contour_density": 0.0,
            }

        h, w = frame_bgr.shape[:2]
        total_pixels = float(h * w)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        sig_edges = mag > 25.0
        sig_count = int(np.count_nonzero(sig_edges))

        if sig_count > 50:
            angles = np.abs(np.arctan2(gy[sig_edges], gx[sig_edges])) * 180.0 / np.pi
            is_horiz = np.abs(angles - 90.0) <= 10.0
            is_vert = (angles <= 10.0) | (angles >= 170.0)
            orthogonal_ratio = float(np.count_nonzero(is_horiz | is_vert) / sig_count)
        else:
            orthogonal_ratio = 0.0

        flat_ratio = float(np.count_nonzero(mag < 10.0) / total_pixels)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        text_like_count = 0
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if 15 <= area <= 4000 and 0.15 <= (bw / max(1, bh)) <= 4.0:
                text_like_count += 1

        text_density = float(text_like_count / (total_pixels / 10000.0))

        return {
            "orthogonal_edge_ratio": orthogonal_ratio,
            "flat_color_ratio": flat_ratio,
            "text_contour_density": text_density,
        }

    def classify(
        self,
        video_path: VideoInput,
        video_name: str | None = None,
    ) -> VideoModality:
        """Classify video file as SCREEN_RECORDING or NATURAL_VIDEO.

        Args:
            video_path: Video file path, stream (BinaryIO), or upload object.
            video_name: Optional explicit name for the video.

        Returns:
            VideoModality enum value.
        """
        target, source_name, _ = resolve_video_input(video_path, video_name=video_name)
        if isinstance(target, str) and not Path(target).is_file():
            logger.error("Video file does not exist: %s", target)
            return VideoModality.UNKNOWN

        try:
            container = open_video_container(target)
        except Exception as exc:
            logger.error("Failed to open video %s: %s", source_name, exc)
            return VideoModality.UNKNOWN

        frames: list[np.ndarray] = []
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            total_frames = stream.frames
            total_sec = None
            if stream.duration and stream.time_base:
                total_sec = float(stream.duration * stream.time_base)
            elif container.duration:
                total_sec = float(container.duration / av.time_base)

            if total_sec and total_sec > 0:
                interval = total_sec / (self.sample_frame_count + 1)
                target_times = [interval * (i + 1) for i in range(self.sample_frame_count)]

                for t in target_times:
                    try:
                        target_pts = int(t / stream.time_base)
                        container.seek(target_pts, stream=stream)
                        for frame in container.decode(video=0):
                            frames.append(frame.to_ndarray(format="bgr24"))
                            break
                    except Exception as seek_err:
                        logger.debug("Seek failed at %.1fs: %s", t, seek_err)
            else:
                step = max(1, total_frames // (self.sample_frame_count + 1)) if total_frames > 0 else 30
                for i, frame in enumerate(container.decode(video=0)):
                    if i % step == 0:
                        frames.append(frame.to_ndarray(format="bgr24"))
                        if len(frames) >= self.sample_frame_count:
                            break

        except Exception as exc:
            logger.warning("Error reading video frames for classification %s: %s", source_name, exc)
        finally:
            container.close()
            rewind_stream(target)

        if not frames:
            return VideoModality.UNKNOWN

        ortho_scores: list[float] = []
        flat_scores: list[float] = []
        text_scores: list[float] = []

        for frame in frames:
            metrics = self.analyze_frame(frame)
            ortho_scores.append(metrics["orthogonal_edge_ratio"])
            flat_scores.append(metrics["flat_color_ratio"])
            text_scores.append(metrics["text_contour_density"])

        sorted_ortho = sorted(ortho_scores)
        sorted_flat = sorted(flat_scores)
        sorted_text = sorted(text_scores)
        idx = int(len(sorted_ortho) * 0.6)
        rep_ortho = sorted_ortho[min(idx, len(sorted_ortho) - 1)]
        rep_flat = sorted_flat[min(idx, len(sorted_flat) - 1)]
        rep_text = sorted_text[min(idx, len(sorted_text) - 1)]

        logger.debug(
            "Modality classification for %s: rep_ortho=%.3f, rep_flat=%.3f, rep_text=%.3f",
            source_name,
            rep_ortho,
            rep_flat,
            rep_text,
        )

        is_screen = (
            (rep_ortho >= self.orthogonal_threshold and rep_flat >= self.flat_color_threshold)
            or (rep_flat >= self.flat_color_threshold and rep_text >= self.text_density_threshold and rep_ortho >= 0.40)
        )
        if is_screen:
            return VideoModality.SCREEN_RECORDING
        return VideoModality.NATURAL_VIDEO
