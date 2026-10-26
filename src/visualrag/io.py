"""I/O utilities for resolving video inputs from files, streams, and web uploads."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, BinaryIO, Union

import av

logger = logging.getLogger(__name__)

# Standard VideoInput type: supports file path, Path object, or any binary stream
VideoInput = Union[str, Path, BinaryIO, io.IOBase, Any]


def resolve_video_input(
    video: VideoInput,
    video_name: str | None = None,
) -> tuple[Any, str, str]:
    """Resolve various video inputs into a PyAV-compatible target, source name, and stem.

    Handles:
    - Filesystem paths (str, Path)
    - Standard library binary streams (io.BufferedReader, io.BytesIO, io.IOBase)
    - Web framework upload wrappers (FastAPI UploadFile, Flask FileStorage, Django UploadedFile)

    Args:
        video: Video file path, stream, or upload object.
        video_name: Optional explicit name for the video.

    Returns:
        A tuple of ``(stream_or_path, source_name, stem)``:
        - ``stream_or_path``: Object to pass to ``av.open(..., mode='r')``.
        - ``source_name``: Human-readable video name / path for metadata citations.
        - ``stem``: Safe filesystem stem (no extension) for naming generated slides/frames.
    """
    # Unwrap framework upload objects if present
    # FastAPI / Starlette UploadFile -> .file
    # Flask / Werkzeug FileStorage -> .stream
    # Django UploadedFile -> .file
    target = getattr(video, "file", getattr(video, "stream", video))

    # Determine human-readable source_name
    if video_name:
        source_name = video_name
    elif hasattr(video, "filename") and isinstance(video.filename, str) and video.filename:
        source_name = video.filename
    elif hasattr(video, "name") and isinstance(video.name, str) and video.name:
        source_name = video.name
    elif isinstance(video, (str, Path)):
        source_name = str(video)
    else:
        source_name = "video.mp4"

    # Determine safe stem (strip extensions and path components)
    stem = Path(source_name).stem or "video"

    # Prepare the stream target
    if isinstance(target, (str, Path)):
        return str(target), source_name, stem

    # Rewind stream to beginning if seekable
    rewind_stream(target)

    return target, source_name, stem


def rewind_stream(stream: Any) -> None:
    """Safely rewind a stream to byte 0 if it supports seek."""
    if hasattr(stream, "seek"):
        try:
            if not hasattr(stream, "seekable") or stream.seekable():
                stream.seek(0)
        except Exception as exc:
            logger.debug("Failed to seek stream to 0: %s", exc)


def open_video_container(target: Any) -> av.container.InputContainer:
    """Open a PyAV container in read mode from a path or binary stream.

    Args:
        target: A file path string, Path object, or seekable BinaryIO stream.

    Returns:
        An open :class:`av.container.InputContainer`.
    """
    if isinstance(target, (str, Path)):
        return av.open(str(target), mode="r")

    # Rewind before opening container
    rewind_stream(target)
    return av.open(target, mode="r")
