"""Deictic intent detection across lexical, syntactic, and semantic stages."""

import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from visualrag.models import DeicticIntent, TranscriptSegment

logger = logging.getLogger(__name__)


class LexicalFilter:
    """Fast regular-expression filter for deictic markers and demonstratives.

    Filters out non-deictic speech with minimal overhead across software UI,
    screen presentations, documents, and physical demonstration videos.
    """

    # Spatial demonstratives, locatives, and screen/spatial regions
    _SPATIAL_DEMONSTRATIVES = re.compile(
        r"\b(?:"
        r"right\s+here|over\s+here|down\s+here|up\s+here|in\s+here|out\s+here|around\s+here|"
        r"right\s+there|over\s+there|down\s+there|up\s+there|in\s+there|around\s+there|"
        r"like\s+this|look\s+here|"
        r"top\s+left|top\s+right|bottom\s+left|bottom\s+right|"
        r"upper\s+left|upper\s+right|lower\s+left|lower\s+right|"
        r"on\s+the\s+(?:left|right|top|bottom)"
        r")\b|"
        r"\b(?:this|these|here|there|that|those)\b",
        re.IGNORECASE,
    )

    # Perception imperatives & observational triggers
    _PERCEPTION_TRIGGERS = re.compile(
        r"\b(?:"
        r"look\s+at|look\s+closely|notice|observe|watch|"
        r"see\s+(?:how|this|that|these|the)|"
        r"check\s+out|view|inspect|zoom\s+in|focus\s+on"
        r")\b",
        re.IGNORECASE,
    )

    # Deictic action verbs spanning software/UI interactions, screen gestures, and physical manipulation
    _ACTION_VERBS = re.compile(
        r"\b(?:"
        # Software & UI interactions
        r"click|double-click|right-click|press|tap|touch|hover|drag|drop|select|"
        r"toggle|highlight|scroll|swipe|type|enter|input|check|uncheck|choose|"
        r"expand|collapse|switch|zoom|"
        # Physical & general manipulations
        r"plug|connect|attach|insert|push|pull|turn|rotate|screw|place|put|mount|hook|grab|pick|open|close"
        r")\b.*?"
        r"\b(?:into|to|on|onto|in|at|here|there|this|that|these|those)\b",
        re.IGNORECASE,
    )

    def matches(self, text: str) -> bool:
        """Check if the text contains any candidate lexical deictic triggers."""
        if not text:
            return False
        return bool(
            self._SPATIAL_DEMONSTRATIVES.search(text)
            or self._PERCEPTION_TRIGGERS.search(text)
            or self._ACTION_VERBS.search(text)
        )


class SyntacticDeicticFilter:
    """Dependency parser filter to isolate physical and screen pointing and object references.

    Uses syntactic dependency analysis to filter out:
    - Conversational fillers ("Look, I think...", "You see, ...")
    - Abstract or sentential demonstratives ("This means...", "Because of this...")
    while preserving concrete references ("Look at this button", "Connect this here", "Hover here").
    """

    _ABSTRACT_VERBS = {
        "mean", "means", "meant",
        "show", "shows", "showed", "shown",
        "imply", "implies", "implied",
        "suggest", "suggests", "suggested",
        "indicate", "indicates", "indicated",
        "prove", "proves", "proved",
        "cause", "causes", "caused",
        "happen", "happens", "happened",
    }

    _CAUSAL_PREPS = {"because", "due", "instead"}

    _SPATIAL_PREV = {
        "right", "over", "down", "up", "in", "around", "back", "out", "near",
        "under", "through", "across",
    }
    _POINTING_VERBS = {
        # Software & UI interaction verbs
        "click", "double-click", "right-click", "press", "tap", "touch", "hover",
        "drag", "drop", "select", "toggle", "highlight", "scroll", "swipe", "type",
        "enter", "choose", "expand", "collapse", "switch", "zoom", "check", "uncheck",
        # General & physical pointing / manipulation verbs
        "point", "show", "plug", "connect", "insert", "mount", "place", "attach",
        "hook", "put", "grab", "pick", "open", "close", "turn", "push", "pull",
    }
    _ABSTRACT_NOUNS = {
        "point", "way", "case", "time", "reason", "thing",
        "morning", "afternoon", "evening", "week", "year", "month",
        "sort", "kind", "type", "part", "sense", "regard", "context", "lab",
    }

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self._spacy_model = spacy_model
        self._nlp = None

    def _get_nlp(self) -> Any:
        if self._nlp is None:
            try:
                import spacy

                try:
                    self._nlp = spacy.load(self._spacy_model)
                except OSError:
                    logger.info("spaCy model '%s' not found locally. Downloading automatically...", self._spacy_model)
                    from spacy.cli import download

                    download(self._spacy_model)
                    self._nlp = spacy.load(self._spacy_model)
            except Exception as exc:
                logger.warning(
                    "spaCy model '%s' unavailable (%s). Falling back to heuristic rules.",
                    self._spacy_model,
                    exc,
                )
                self._nlp = False
        return self._nlp

    def filter(self, text: str) -> bool:
        """Evaluate if text contains valid exophoric deixis.

        Returns:
            True if text contains an exophoric deictic act, False if purely
            discourse filler or abstract pronoun.
        """
        # Reject forward-looking future promises: "look at that in a minute", "see that later"
        if re.search(r"\b(?:in\s+a\s+(?:minute|second|moment|bit)|later|coming\s+up|soon)\b", text, re.IGNORECASE):
            return False

        nlp = self._get_nlp()
        if not nlp:
            # Heuristic fallback if spaCy is not available
            return self._heuristic_fallback(text)

        doc = nlp(text)

        has_exophoric = False

        for token in doc:
            token_lower = token.text.lower()

            # Rule 1: Perception verbs ("look", "see", "notice", "watch", "view", "inspect")
            if token_lower in {"look", "see", "notice", "watch", "view", "inspect"}:
                children = list(token.children)
                child_deps = {c.dep_ for c in children}
                preps = [c for c in children if c.dep_ == "prep" and c.text.lower() in {"at", "into", "in", "to", "on"}]
                dobj = [c for c in children if c.dep_ in {"dobj", "pobj"}]
                if preps or dobj:
                    has_exophoric = True
                elif any(c.text == "," for c in children) or token.dep_ in {"advmod", "discourse", "intj", "parataxis"}:
                    continue

            # Rule 2: Concrete demonstratives ("this", "these")
            if token_lower in {"this", "these"}:
                if token.dep_ == "det" and token.head.pos_ in {"NOUN", "PROPN"}:
                    head_lemma = token.head.lemma_.lower()
                    if head_lemma not in self._ABSTRACT_NOUNS:
                        has_exophoric = True

                elif token.dep_ in {"nsubj", "nsubjpass"}:
                    head_verb = token.head.lemma_.lower()
                    if head_verb in self._ABSTRACT_VERBS:
                        continue
                    if token.head.text.lower() in self._CAUSAL_PREPS:
                        continue
                    # Check next tokens for abstract connectors ("this is why...", "this is what...")
                    next_tokens = [doc[i].text.lower() for i in range(token.i + 1, min(len(doc), token.i + 4))]
                    if any(t in {"what", "why", "because", "how", "means"} for t in next_tokens):
                        continue
                    has_exophoric = True

                elif token.dep_ in {"dobj", "pobj"}:
                    head_lower = token.head.text.lower()
                    head_head_lower = token.head.head.text.lower() if token.head.head != token.head else ""
                    if head_lower in self._CAUSAL_PREPS or head_head_lower in self._CAUSAL_PREPS:
                        continue
                    has_exophoric = True

            # Rule 3: Spatial locatives ("right here", "over here", "down here", "up here", "in here", "hover here")
            if token_lower in {"here", "there"}:
                prev_token = doc[token.i - 1] if token.i > 0 else None
                if prev_token and prev_token.text.lower() in self._SPATIAL_PREV:
                    has_exophoric = True
                elif token.head.lemma_.lower() in self._POINTING_VERBS:
                    has_exophoric = True

            # Rule 4: Demonstratives "that", "those"
            if token_lower in {"that", "those"}:
                # Discourse anaphora rejection: "that's what", "that's why", "that's because", "that is why"
                if token.dep_ in {"nsubj", "nsubjpass"}:
                    next_tokens = [doc[i].text.lower() for i in range(token.i + 1, min(len(doc), token.i + 4))]
                    if any(t in {"what", "why", "because", "how", "exactly", "means"} for t in next_tokens):
                        continue
                if token.dep_ == "det" and token.head.pos_ in {"NOUN", "PROPN"}:
                    if token.head.lemma_.lower() not in self._ABSTRACT_NOUNS:
                        has_exophoric = True
                elif token.dep_ in {"dobj", "pobj"}:
                    head_verb = token.head.lemma_.lower() if token.head.pos_ == "VERB" else token.head.head.lemma_.lower()
                    if head_verb in self._POINTING_VERBS or head_verb in {
                        "look", "see", "watch", "check", "notice", "grab", "take", "plug"
                    }:
                        has_exophoric = True

        return has_exophoric

    def _heuristic_fallback(self, text: str) -> bool:
        """Regex/string heuristic when spaCy pipeline is not loaded."""
        cleaned = text.strip()
        # Reject obvious fillers and discourse anaphora
        if re.match(r"^(?:look|see|you\s+see)[,\s]+", cleaned, re.IGNORECASE):
            if not re.search(r"\b(?:look\s+at|see\s+(?:the|this|these|how))\b", cleaned, re.IGNORECASE):
                return False
        if re.match(r"^(?:this|that)\s+(?:is\s+why|means|shows|implies|suggests|is\s+because)\b", cleaned, re.IGNORECASE):
            return False
        if re.search(r"\b(?:that's|that\s+is)\s+(?:what|why|because|how|exactly)\b", cleaned, re.IGNORECASE):
            return False
        if re.search(r"\bbecause\s+of\s+(?:this|that)\b", cleaned, re.IGNORECASE):
            return False
        if re.search(r"\bhere\s+to\s+see\b", cleaned, re.IGNORECASE):
            return False
        return True


class DeicticDetector:
    """Multi-stage visual deictic intent detector.

    Combines lexical pattern matching, syntactic dependency parsing, and
    an optional language model to determine whether a spoken utterance references
    a visual, spatial, software UI, or physical item visible in the scene.
    """

    SYSTEM_PROMPT = (
        "You are an expert audio-visual deictic intent analyzer for video understanding.\n"
        "Your task is to determine whether the speaker in the transcript is referring to a visual or spatial "
        "entity, action, or area currently visible in the video scene (exophoric deixis).\n\n"
        "Criteria for is_visual_deictic=True:\n"
        "- The utterance directs visual attention: 'look at this section', 'notice the icon in the corner', 'see this diagram'.\n"
        "- The utterance describes a software or UI interaction: 'click this button', 'hover here', 'drag this into that folder', 'tap here', 'toggle this switch'.\n"
        "- The utterance describes physical manipulation or pointing: 'plug this in here', 'mount this bracket', 'turn this knob'.\n"
        "- The utterance references a visual data point, reading, or component: 'this value is 45', 'this line on the chart shows the spike'.\n\n"
        "Criteria for is_visual_deictic=False:\n"
        "- Conversational discourse markers / idioms: 'Look, I think that is wrong', 'You see, what happened was', 'Now, look here...'.\n"
        "- Purely abstract or sentential references: 'This means we are on schedule', 'Because of this outcome', 'That is why we did it'.\n"
        "- Future or past promises not visible now: 'We will look at that in a minute', 'As we saw yesterday'.\n\n"
        "When is_visual_deictic=True:\n"
        "1. Extract target_phrase: the specific visual element, UI component, area, or physical entity being referenced.\n"
        "2. Formulate grounding_question: a concise visual question to ask a Multimodal VLM "
        "(e.g. 'What button or UI element is being clicked?', 'What component is being connected?')."
    )

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        """Initialize the deictic intent detector.

        Args:
            llm: Optional LangChain chat model with structured output capability.
            spacy_model: spaCy English pipeline model name for dependency parsing.
        """
        self.lexical_filter = LexicalFilter()
        self.syntactic_filter = SyntacticDeicticFilter(spacy_model=spacy_model)
        self.llm = llm
        self._structured_llm = None
        if self.llm is not None and hasattr(self.llm, "with_structured_output"):
            try:
                self._structured_llm = self.llm.with_structured_output(DeicticIntent)
            except Exception as exc:
                logger.debug("Model does not support native with_structured_output: %s", exc)
                self._structured_llm = None

    def detect(self, segment: TranscriptSegment) -> DeicticIntent:
        """Run deictic detection pipeline on a transcript segment.

        Args:
            segment: Spoken transcript segment.

        Returns:
            DeicticIntent with decision, target phrase, and grounding question.
        """
        text = segment.text.strip()

        # Lexical pattern pre-filtering
        if not self.lexical_filter.matches(text):
            return DeicticIntent(
                is_visual_deictic=False,
                confidence=0.99,
                target_phrase=None,
                grounding_question=None,
            )

        # Syntactic dependency filtering
        if not self.syntactic_filter.filter(text):
            return DeicticIntent(
                is_visual_deictic=False,
                confidence=0.95,
                target_phrase=None,
                grounding_question=None,
            )

        # Semantic verification via structured model if configured
        if self._structured_llm is not None:
            try:
                prompt_text = f"Utterance: \"{text}\""
                intent: DeicticIntent = self._structured_llm.invoke([
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=prompt_text),
                ])
                if intent.is_visual_deictic:
                    target_phrase = intent.target_phrase or self._extract_heuristic_target(text)
                    question = intent.grounding_question or self._build_grounding_question(text, target_phrase)
                    return DeicticIntent(
                        is_visual_deictic=True,
                        confidence=intent.confidence,
                        target_phrase=target_phrase,
                        grounding_question=question,
                    )
                return intent
            except Exception as exc:
                logger.warning("LLM structured output invocation failed: %s", exc)

        elif self.llm is not None:
            try:
                prompt_text = (
                    "You are an expert audio-visual deictic classifier.\n"
                    "Determine whether the speaker is pointing at, demonstrating, or referring to a visible element, "
                    "UI control, screen component, or physical object in the scene.\n"
                    f"Utterance: \"{text}\"\n"
                    "Respond ONLY with 'YES' or 'NO'."
                )
                res = self.llm.invoke([HumanMessage(content=prompt_text)])
                ans = str(res.content if hasattr(res, "content") else res).strip()
                ans_lower = ans.strip().lower()
                first_word = re.split(r"[\s,.:;!]+", ans_lower)[0] if ans_lower else ""
                is_visual = (
                    first_word == "yes"
                    or ("yes" in ans_lower and "no" not in ans_lower)
                    or ("visible" in ans_lower and "not" not in ans_lower)
                )
                target_phrase = self._extract_heuristic_target(text)
                return DeicticIntent(
                    is_visual_deictic=is_visual,
                    confidence=1.0,
                    target_phrase=target_phrase if is_visual else None,
                    grounding_question=self._build_grounding_question(text, target_phrase) if is_visual else None,
                )
            except Exception as exc:
                logger.warning("Plain LLM invocation failed (%s). Falling back to heuristic rules.", exc)
        target_phrase = self._extract_heuristic_target(text)
        return DeicticIntent(
            is_visual_deictic=True,
            confidence=0.85,
            target_phrase=target_phrase,
            grounding_question=self._build_grounding_question(text, target_phrase),
        )

    def _build_grounding_question(self, text: str, target_phrase: str | None) -> str:
        """Formulate a specific, context-rich grounding question for the downstream VLM."""
        target = target_phrase or self._extract_heuristic_target(text)
        clean_utterance = text.strip().rstrip(".?!")
        if not target or target.lower() in {"that", "this", "these", "those", "it", "visual element"}:
            noun_phrase = "referenced visual element"
        else:
            noun_phrase = target
        return (
            f"The speaker notes: '{clean_utterance}'. "
            f"Describe the appearance, position, and specific visual details of the {noun_phrase} shown in this scene."
        )

    def _extract_heuristic_target(self, text: str) -> str:
        """Extract candidate target phrase from text when LLM structured output is unavailable."""
        cleaned = text.strip()
        # Demonstrative equative constructions ('this/that is ...')
        m1 = re.search(
            r"\b(?:this|that)\s+is\s+(?:an?|the)?\s*([a-zA-Z0-9_\-\s]{2,40}?)(?=$|[,\.\?!]|\s+(?:with|that|which|in|on|where))\b",
            cleaned,
            re.IGNORECASE,
        )
        if m1 and m1.group(1).strip():
            return m1.group(1).strip()

        # Perception imperative targets ('look at ...', 'see ...')
        m2 = re.search(
            r"\b(?:look\s+at|notice|see|closer\s+look\s+at)\s+(?:the|this|these|those)?\s*([a-zA-Z0-9_\-\s]{2,40}?)(?=$|[,\.\?!]|\s+(?:here|there|is|on|in|with))\b",
            cleaned,
            re.IGNORECASE,
        )
        if m2 and m2.group(1).strip():
            return m2.group(1).strip()

        # Action verb targets ('click ...', 'hover over ...')
        m3 = re.search(
            r"\b(?:click|press|tap|touch|hover|drag|drop|select|toggle|choose)\s+(?:on|over|onto)?\s*(?:the|this|that|these|those)?\s*([a-zA-Z0-9_\-\s]{2,35}?)(?=$|[,\.\?!]|\s+(?:here|there|to|into|on|in|with))\b",
            cleaned,
            re.IGNORECASE,
        )
        if m3 and m3.group(1).strip() and m3.group(1).strip().lower() not in {"here", "there", "this", "that"}:
            return m3.group(1).strip()

        # Demonstrative determiner phrases ('this/that/these/those ...')
        m4 = re.search(
            r"\b(?:this|these|that|those)\s+([a-zA-Z0-9_\-\s]{2,35}?)(?=$|[,\.\?!]|\s+(?:is|to|into|on|here|there|was|controls|starts|right))\b",
            cleaned,
            re.IGNORECASE,
        )
        if m4 and m4.group(1).strip():
            return m4.group(1).strip()

        return "visual element"
