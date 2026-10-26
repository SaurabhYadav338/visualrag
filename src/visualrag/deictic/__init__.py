"""Deictic Vision engine for camera and natural video understanding."""

from visualrag.deictic.detector import (
    DeicticDetector,
    LexicalFilter,
    SyntacticDeicticFilter,
)
from visualrag.deictic.extractor import DeicticExtractor

__all__ = [
    "DeicticDetector",
    "DeicticExtractor",
    "LexicalFilter",
    "SyntacticDeicticFilter",
]
