"""Multi-level deduplication engine (temporal sliding window + pHash + semantic co-reference).

Designed for production use across diverse video domains:
- Software screencasts & UI demos (buttons, toggles, menus, tabs, panels)
- Hardware & electronics engineering (wires, pins, boards, scopes, components)
- Manufacturing & field operations (valves, pipes, gauges, pumps, switches)
- Healthcare & medical training (instruments, anatomical landmarks, monitors)
- Whiteboard, education & financial presentations (charts, formulas, slides)
"""

from __future__ import annotations

import logging
import re

import imagehash

from visualrag.models import DeicticEvent

logger = logging.getLogger(__name__)


class DeicticDeduplicator:
    """Deduplicates visual deictic events across temporal, visual, and semantic axes.

    Prevents duplicate keyframe extractions and redundant visual anchors when a speaker
    repeatedly references or gestures toward the same visual entity across successive sentences.
    Evaluates:
    - Temporal proximity within a bounded sliding window.
    - Visual similarity via perceptual hash (pHash) Hamming distance.
    - Semantic co-reference with morphological normalization and discriminator conflict resolution.
    """

    # Demonstrative and filler tokens that indicate pronominal anaphora rather than a new entity
    _STOP_WORDS = {
        "the", "a", "an", "this", "these", "that", "those",
        "here", "there", "it", "its", "them", "their", "they",
        "one", "ones", "item", "object", "visual", "element", "component",
        "thing", "things", "part", "section", "area", "something", "stuff",
        "look", "see", "notice", "watch", "check", "view",
    }

    # Color discriminators: two references with different colors are distinct entities
    _COLORS = {
        "red", "blue", "green", "yellow", "orange", "purple",
        "black", "white", "gray", "grey", "brown", "pink", "cyan", "magenta",
    }

    # Polar / directional opposites that distinguish adjacent controls or components
    _POLAR_OPPOSITES: tuple[frozenset[str], ...] = (
        frozenset({"left", "right"}),
        frozenset({"top", "bottom"}),
        frozenset({"upper", "lower"}),
        frozenset({"front", "back"}),
        frozenset({"in", "out"}),
        frozenset({"inside", "outside"}),
        frozenset({"inner", "outer"}),
        frozenset({"input", "output"}),
        frozenset({"inlet", "outlet"}),
        frozenset({"start", "stop"}),
        frozenset({"on", "off"}),
        frozenset({"open", "closed"}),
        frozenset({"horizontal", "vertical"}),
        frozenset({"first", "second", "third", "fourth", "fifth", "last"}),
    )

    def __init__(
        self,
        dedup_window_sec: float = 45.0,
        max_hash_distance: int = 5,
        max_history_size: int = 150,
    ) -> None:
        """Initialize the deictic event deduplicator.

        Args:
            dedup_window_sec: Temporal sliding window in seconds; events older than this
                threshold expire from deduplication consideration.
            max_hash_distance: Maximum perceptual hash Hamming distance for visual equivalence
                (for 64-bit hashes, 5 corresponds to ~92% visual match).
            max_history_size: Maximum number of events retained in memory to prevent unbounded
                growth during long video streams.
        """
        self.dedup_window_sec = dedup_window_sec
        self.max_hash_distance = max_hash_distance
        self.max_history_size = max(1, max_history_size)
        self._history: list[DeicticEvent] = []

    def reset(self) -> None:
        """Clear all deduplication history and reset state between video sessions."""
        self._history.clear()

    def check(
        self,
        timestamp_sec: float,
        image_hash: str,
        target_phrase: str | None,
        source_video: str | None = None,
    ) -> tuple[bool, DeicticEvent | None]:
        """Check if the candidate event matches a recent cached event.

        Args:
            timestamp_sec: Timestamp of candidate event in seconds.
            image_hash: Hexadecimal perceptual hash string of candidate keyframe.
            target_phrase: Identified visual target entity or spoken noun phrase.
            source_video: Optional source video identifier to avoid cross-video deduplication.

        Returns:
            Tuple of (is_duplicate: bool, matching_cached_event: DeicticEvent | None).
        """
        if not self._history or not image_hash:
            return False, None

        try:
            cand_hash = imagehash.hex_to_hash(image_hash)
        except Exception:
            return False, None

        # Clean old events outside sliding temporal window (guarded against negative jumps)
        self._history = [
            ev for ev in self._history
            if 0.0 <= (timestamp_sec - ev.timestamp_sec) <= self.dedup_window_sec
        ]

        # Scan recent events in reverse chronological order
        for cached_ev in reversed(self._history):
            # Source video isolation
            if source_video:
                cached_src = cached_ev.metadata.get("source_video")
                if cached_src and cached_src != source_video:
                    continue

            # Temporal sliding window boundary
            dt = timestamp_sec - cached_ev.timestamp_sec
            if dt < 0 or dt > self.dedup_window_sec:
                continue

            # Perceptual hash Hamming distance threshold
            try:
                cached_hash = imagehash.hex_to_hash(cached_ev.image_hash)
                dist = cand_hash - cached_hash
                if dist > self.max_hash_distance:
                    continue
            except Exception:
                continue

            # Multi-domain semantic co-reference resolution
            if self._is_coreferent(cached_ev.target_phrase, target_phrase):
                logger.debug(
                    "Deduplication hit at %.1fs with cached event at %.1fs (pHash dist=%d, phrase='%s' ~ '%s')",
                    timestamp_sec,
                    cached_ev.timestamp_sec,
                    dist,
                    target_phrase,
                    cached_ev.target_phrase,
                )
                return True, cached_ev

        return False, None

    def register(self, event: DeicticEvent) -> None:
        """Register a newly processed deictic event in the deduplication history."""
        self._history.append(event)
        if len(self._history) > self.max_history_size:
            self._history = self._history[-self.max_history_size:]

    @classmethod
    def _stem_word(cls, word: str) -> str:
        """Fast morphological suffix normalization for English nouns and verbs."""
        w = word.lower()
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 4 and w[-3] in {"s", "h", "x", "z"}:
            return w[:-2]
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            return w[:-1]
        return w

    @classmethod
    def _extract_tokens(cls, phrase: str | None) -> list[str]:
        """Extract alphanumeric tokens from a target phrase."""
        if not phrase:
            return []
        return [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-]+\b", phrase)]

    @classmethod
    def _extract_content_stems(cls, phrase: str | None) -> set[str]:
        """Extract stemmed content tokens with stop words removed."""
        tokens = cls._extract_tokens(phrase)
        return {
            cls._stem_word(t) for t in tokens
            if t not in cls._STOP_WORDS and cls._stem_word(t) not in cls._STOP_WORDS
        }

    @classmethod
    def _has_conflicting_discriminators(cls, tokens_a: list[str], tokens_b: list[str]) -> bool:
        """Detect conflicting modifiers (e.g. 'red' vs 'blue', 'pin 1' vs 'pin 2', 'left' vs 'right')."""
        set_a = set(tokens_a)
        set_b = set(tokens_b)

        # Conflicting color attributes
        colors_a = set_a & cls._COLORS
        colors_b = set_b & cls._COLORS
        if colors_a and colors_b and colors_a != colors_b:
            return True

        # Conflicting polar opposites
        for pair in cls._POLAR_OPPOSITES:
            if (set_a & pair) and (set_b & pair) and (set_a & pair) != (set_b & pair):
                return True

        # Conflicting numeric identifiers (e.g., '1' vs '2', 'step 3' vs 'step 4')
        nums_a = {t for t in set_a if t.isdigit()}
        nums_b = {t for t in set_b if t.isdigit()}
        if nums_a and nums_b and nums_a != nums_b:
            return True

        return False

    def _is_coreferent(self, cached_phrase: str | None, new_phrase: str | None) -> bool:
        """Evaluate if two target phrases refer to the same visual entity across domains."""
        # Clean string values
        c_raw = (cached_phrase or "").strip()
        n_raw = (new_phrase or "").strip()

        # Extract tokens and content stems
        c_tokens = self._extract_tokens(c_raw)
        n_tokens = self._extract_tokens(n_raw)
        c_stems = self._extract_content_stems(c_raw)
        n_stems = self._extract_content_stems(n_raw)

        # Pronominal anaphora rule:
        # If candidate utterance is purely demonstrative / pronominal ("this", "it", "that", "here"),
        # it binds to the active visual anchor in the current stationary scene.
        if not n_stems:
            return True

        # If previous event had no specific noun (was generic like "look here"),
        # but the candidate explicitly introduces a distinct named entity, do NOT deduplicate.
        if not c_stems and n_stems:
            return False

        # Reject if conflicting discriminators are present (e.g. red wire vs blue wire, pin 1 vs pin 2)
        if self._has_conflicting_discriminators(c_tokens, n_tokens):
            return False

        # Intersection of stemmed content words (e.g. "red wire" and "wire", "resistors" and "resistor")
        overlap = c_stems & n_stems
        if overlap:
            return True

        # Substring / compound containment (e.g. "arduino" in "arduino nano board", "submit" in "submit button")
        c_clean = " ".join(sorted(c_stems))
        n_clean = " ".join(sorted(n_stems))
        if c_clean and n_clean and (c_clean in n_clean or n_clean in c_clean):
            return True

        # Jaccard similarity threshold for longer technical phrases
        union = c_stems | n_stems
        if union and (len(overlap) / len(union)) >= 0.5:
            return True

        return False

