"""Optical character recognition for screen frames using Tesseract."""

import io
import logging
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

from visualrag.models import ScreenFrame

logger = logging.getLogger(__name__)


def extract_text(image_input: Path | str | bytes | np.ndarray | Image.Image) -> tuple[str, float]:
    """Run Tesseract OCR on an image and compute average word confidence.

    Args:
        image_input: File path, raw bytes (PNG/JPEG), NumPy array (BGR or grayscale),
            or PIL Image.

    Returns:
        A tuple of (extracted_text, mean_confidence) where confidence is between
        0.0 and 100.0. Returns ("", 0.0) on failure or empty input.
    """
    try:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                logger.warning("Image file does not exist: %s", path)
                return "", 0.0
            img = Image.open(path)
        elif isinstance(image_input, bytes):
            if not image_input:
                return "", 0.0
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                return "", 0.0
            if len(image_input.shape) == 3:
                img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            logger.warning("Unsupported image input type for OCR: %s", type(image_input))
            return "", 0.0

        data = pytesseract.image_to_data(img, output_type=Output.DICT)

        words = []
        confidences = []
        for text, conf in zip(data["text"], data["conf"]):
            if str(conf) == "-1" or conf == -1:
                continue
            cleaned = str(text).strip()
            if not cleaned:
                continue
            words.append(cleaned)
            confidences.append(float(conf))

        extracted = " ".join(words)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return extracted, mean_conf

    except Exception as exc:
        logger.warning("OCR extraction failed for %s: %s", type(image_input), exc)
        return "", 0.0


def enrich_frames_with_ocr(screens: list[ScreenFrame]) -> list[ScreenFrame]:
    """Extract text for all screen frames in-place from their in-memory image bytes."""
    for screen in screens:
        if screen.image_bytes:
            screen.ocr_text, screen.ocr_confidence = extract_text(screen.image_bytes)
    return screens
