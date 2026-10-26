"""Temporal alignment of audio transcript segments with extracted screen frames."""

import logging

from visualrag.models import ScreenFrame, TranscriptSegment

logger = logging.getLogger(__name__)


def align_transcript(
    segments: list[TranscriptSegment],
    screens: list[ScreenFrame],
) -> list[ScreenFrame]:
    """Map transcript segments to screen frames based on time interval overlap.

    For each segment, finds the screen frame displayed during that time window and
    appends the spoken text to `ScreenFrame.transcript_segment`. If an utterance
    spans a transition, it is assigned to the screen frame with the greatest temporal overlap.

    Args:
        segments: Timestamped transcript segments.
        screens: ScreenFrame objects (mutated in-place).

    Returns:
        The mutated list of screen frames.
    """
    if not segments or not screens:
        return screens

    # Support flexible argument ordering (segments, screens) or (screens, segments)
    if segments and isinstance(segments[0], ScreenFrame):
        segments, screens = screens, segments  # type: ignore[assignment]

    sorted_screens = sorted(screens, key=lambda s: s.start_time_sec)

    for seg in segments:
        best_screen: ScreenFrame | None = None
        best_overlap: float = 0.0

        for screen in sorted_screens:
            effective_end = screen.end_time_sec if screen.end_time_sec is not None else float("inf")

            # Calculate interval overlap
            overlap_start = max(seg.start_sec, screen.start_time_sec)
            overlap_end = min(seg.end_sec, effective_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_screen = screen

        if best_screen is None:
            logger.debug(
                "No screen matched for transcript segment [%.1fs, %.1fs]: %r",
                seg.start_sec,
                seg.end_sec,
                seg.text[:50],
            )
            continue

        if best_screen.transcript_segment:
            best_screen.transcript_segment += "\n" + seg.text
        else:
            best_screen.transcript_segment = seg.text

    return screens

