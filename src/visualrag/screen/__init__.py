"""Screen-share and presentation slide extraction engine."""

from visualrag.screen.aligner import align_transcript
from visualrag.screen.cropper import DynamicCropper
from visualrag.screen.extractor import ScreenExtractor
from visualrag.screen.ocr import enrich_frames_with_ocr, extract_text

__all__ = [
    "DynamicCropper",
    "ScreenExtractor",
    "align_transcript",
    "enrich_frames_with_ocr",
    "extract_text",
]
