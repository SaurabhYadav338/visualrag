"""Screen-share bounding box detection and stabilization."""

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DynamicCropper:
    """Detects and isolates active screen-share regions in video frames.

    Combines grayscale and HSV saturation edge detection to handle both
    light and dark themes. Detected bounding boxes are smoothed across
    frames using an Exponential Moving Average (EMA), and state transitions
    between active sharing and full canvas are debounced to avoid flicker.

    Heuristics:
    - Interior edge density: Distinguishes presentation slides (mostly uniform
      backgrounds with text/diagrams) from participant webcam grids (high density
      of facial edges and tile borders).
    - Anchor stabilization: Prevents the crop box from collapsing into interior
      diagram boxes or floating speaker thumbnails during active screen share.
    - Session reset: Discards stale EMA history whenever screensharing resumes
      after a break.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.25,
        max_interior_edge_density: float = 0.10,
        ema_alpha: float = 0.3,
        debounce_count: int = 3,
        max_area_ratio: float = 1.0,
    ) -> None:
        """Initialize the dynamic slide cropper.

        Args:
            min_area_ratio: Minimum screen fraction a detected contour must occupy.
            max_interior_edge_density: Maximum interior edge ratio before rejecting as a webcam grid.
            ema_alpha: Smoothing factor for Exponential Moving Average on bounding box coordinates.
            debounce_count: Number of consecutive agreeing frames required to switch state.
            max_area_ratio: Maximum allowable area fraction before rejecting as full canvas.
        """
        self._min_area_ratio = min_area_ratio
        self._max_interior_edge_density = max_interior_edge_density
        self._ema_alpha = ema_alpha
        self._debounce_count = debounce_count
        self._max_area_ratio = max_area_ratio

        self._prev_bbox: tuple[int, int, int, int] | None = None
        self._state: Literal["on", "off"] = "off"
        self._pending_state: Literal["on", "off"] = "off"
        self._pending_count: int = 0

    def process_frame(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Analyze a BGR frame and return the screen-share region if active.

        Args:
            frame: A BGR image array (H, W, 3).

        Returns:
            Smoothed (x, y, w, h) bounding box, or None if no active presentation is found.
        """
        raw_bbox = self._detect(frame)
        new_state: Literal["on", "off"] = "on" if raw_bbox is not None else "off"

        # Debounce state transitions
        prev_confirmed = self._state
        if new_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = new_state
            self._pending_count = 1

        if self._pending_count >= self._debounce_count:
            self._state = self._pending_state

        # Discard stale smoothing history on a new sharing session
        if prev_confirmed == "off" and self._state == "on":
            self._prev_bbox = None

        if self._state == "off":
            return None

        if raw_bbox is None:
            # Still in "on" state during debounce; hold previous box
            return self._prev_bbox

        return self._smooth(raw_bbox)

    def reset(self) -> None:
        """Reset internal tracking state and smoothing history."""
        self._prev_bbox = None
        self._state = "off"
        self._pending_state = "off"
        self._pending_count = 0

    def _detect(
        self, frame: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        h, w = frame.shape[:2]
        frame_area = h * w

        # Pad 1px so windows flush against the video border produce closed contours
        padded_frame = cv2.copyMakeBorder(
            frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        gray = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2GRAY)
        edges_gray = cv2.Canny(gray, 50, 150)

        hsv = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        edges_sat = cv2.Canny(sat, 50, 150)

        edges_raw_padded = cv2.bitwise_or(edges_gray, edges_sat)
        edges_raw = edges_raw_padded[1:-1, 1:-1]

        # Ignore essentially blank frames
        if np.count_nonzero(edges_raw[2:-2, 2:-2]) < 100:
            return None

        # Dilate to bridge video compression artifacts
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges_raw_padded, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_bbox: tuple[int, int, int, int] | None = None
        best_area: float = 0.0

        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue

            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Accept approximately rectangular shapes
            if not (4 <= len(approx) <= 8):
                continue

            bx, by, bw, bh = cv2.boundingRect(approx)
            if bw * bh == 0:
                continue

            x = max(0, bx - 1)
            y = max(0, by - 1)
            orig_bw = min(bw, w - x)
            orig_bh = min(bh, h - y)
            rect_area = orig_bw * orig_bh
            if rect_area == 0:
                continue

            # Standard presentation aspect ratios (4:3 to 16:9 plus margins)
            aspect = orig_bw / orig_bh
            if not (1.2 <= aspect <= 2.2):
                continue

            # Extent check: reject thin/concave polygons
            cnt_area = cv2.contourArea(approx)
            if (cnt_area / (bw * bh)) <= 0.85:
                continue

            # Size thresholds relative to full screen
            if rect_area <= self._min_area_ratio * frame_area or rect_area > self._max_area_ratio * frame_area:
                continue

            # Reject dense grids (e.g., webcam galleries)
            roi_edges = edges_raw[y : y + orig_bh, x : x + orig_bw]
            edge_density = float(np.count_nonzero(roi_edges)) / roi_edges.size
            if edge_density > self._max_interior_edge_density:
                logger.debug(
                    "Rejected contour with edge density %.3f > %.3f (probable camera gallery)",
                    edge_density,
                    self._max_interior_edge_density,
                )
                continue

            if rect_area > best_area:
                best_area = rect_area
                best_bbox = (x, y, orig_bw, orig_bh)

        # Anti-shrink anchor: prevent collapse into sub-elements when sharing is active
        if self._state == "on" and self._prev_bbox is not None:
            prev_area = self._prev_bbox[2] * self._prev_bbox[3]
            if best_bbox is not None and best_area < 0.50 * prev_area:
                logger.debug("Candidate area < 50% of previous slide; retaining previous anchor")
                best_bbox = self._prev_bbox
            elif best_bbox is None:
                px, py, pw, ph = self._prev_bbox
                roi_prev = edges_raw[py : py + ph, px : px + pw]
                if roi_prev.size > 0:
                    prev_density = float(np.count_nonzero(roi_prev)) / roi_prev.size
                    if 0.01 <= prev_density <= self._max_interior_edge_density:
                        best_bbox = self._prev_bbox

        return best_bbox

    def _smooth(
        self, raw: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        if self._prev_bbox is None:
            self._prev_bbox = raw
            return raw

        alpha = self._ema_alpha
        smoothed = tuple(
            int(alpha * r + (1 - alpha) * p)
            for r, p in zip(raw, self._prev_bbox)
        )
        self._prev_bbox = smoothed  # type: ignore[assignment]
        return smoothed  # type: ignore[return-value]
