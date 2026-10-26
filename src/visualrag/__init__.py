"""visualrag - Extract visual context from meeting recordings and physical demonstration videos."""

from visualrag.screen.aligner import align_transcript
from visualrag.classifier import VideoModalityClassifier
from visualrag.screen.cropper import DynamicCropper
from visualrag.deictic.detector import (
    DeicticDetector,
    LexicalFilter,
    SyntacticDeicticFilter,
)
from visualrag.deictic.extractor import DeicticExtractor
from visualrag.screen.extractor import ScreenExtractor
from visualrag.models import (
    DeicticEvent,
    DeicticIntent,
    ScreenFrame,
    TranscriptSegment,
    TranscriptWord,
    VideoModality,
    VisualSegment,
)
from visualrag.io import VideoInput, resolve_video_input
from visualrag.screen.ocr import enrich_frames_with_ocr, extract_text
from visualrag.pipeline import VisualRAG
from visualrag.storage import (
    CallbackStorage,
    LocalStorage,
    MemoryStorage,
    StorageBackend,
)

__version__ = "0.1.0.dev1"

__all__ = [
    "__version__",
    "CallbackStorage",
    "DeicticDetector",
    "DeicticEvent",
    "DeicticExtractor",
    "DeicticIntent",
    "DynamicCropper",
    "LexicalFilter",
    "LocalStorage",
    "MemoryStorage",
    "ScreenExtractor",
    "ScreenFrame",
    "StorageBackend",
    "SyntacticDeicticFilter",
    "TranscriptSegment",
    "TranscriptWord",
    "VideoInput",
    "VideoModality",
    "VideoModalityClassifier",
    "VisualRAG",
    "VisualSegment",
    "align_transcript",
    "enrich_frames_with_ocr",
    "extract_text",
]
