# VisualRAG

[![PyPI version](https://img.shields.io/pypi/v/visualrag.svg)](https://pypi.org/project/visualrag/)
[![Python Version](https://img.shields.io/pypi/pyversions/visualrag.svg)](https://pypi.org/project/visualrag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A high-performance visual context extraction library for multimodal Video RAG pipelines.**

---

## Overview

Traditional video RAG architectures rely almost exclusively on audio transcription. However, in technical meetings, product walkthroughs and physical demonstration videos, a significant portion of critical knowledge is purely visual:

- **Meeting Recordings & Presentations**: Presenters showcase architecture diagrams, live code, product demos, terminal sessions and dashboards while speaking in indexical shorthand (*"as you can see here..."*, *"this chart on the right"*).
- **Physical Demonstrations & Natural Videos**: Engineers and instructors manipulate hardware, wiring, test equipment, and components (*"connect this wire to that header"*).

`visualrag` extracts, crops, and temporally aligns these high-value visual states with spoken dialogue. Instead of ingesting arbitrary frame dumps, `visualrag` isolates active presentation regions, filters out conference UI clutter (webcams, gallery tiles, sidebars), detects meaningful visual transitions, and emits clean, structured **`VisualSegment`** objects ready for ingestion into any vector database or multimodal LLM agent.

```
    ┌──────────────────────┐         ┌────────────────────────┐
    │     Video Stream     │         │  ASR Audio Transcript  │
    └──────────┬───────────┘         └───────────┬────────────┘
               │                                 │
               ▼                                 │
     [Modality Classifier]                       │
     (Screen vs. Natural)                        │
               │                                 │
      ┌────────┴───────────────┐                 │
      │                        │                 │
      ▼ [If Screen]            ▼ [If Natural]    │
┌──────────────────────┐  ┌──────────────────────┴─┐
│   Screen Extractor   │  │   Deictic Extractor    │
│ • Dynamic UI Cropper │  │ • Lexical/Syntactic    │
│ • PTS Sampling       │  │   spaCy Parser         │
│ • Transition Cooldown│  │ • Kinematic Alignment  │
│ • pHash Dedup        │  │ • Multi-tier Dedup     │
│ • Optional OCR       │  │ • Optional BYOM Verify │
└──────────┬───────────┘  └──────────┬─────────────┘
           │                         │
           ├─────────────────────────┘
           │ (Transcript Aligned to Active Windows)
           ▼
 ┌───────────────────┐
 │   VisualSegment   │ ◄─── Timestamps, Image URI, OCR Text,
 └─────────┬─────────┘      Aligned Spoken Dialogue, Metadata
           │
           ▼
[Vector DB & Agent Pipeline]
(Transcript Search + On-Demand Visual Tool Retrieval)
```

---

## Key Capabilities

### 1. Adaptive Screen & Slide Extraction
- **Dynamic UI Cropper**: Computes the union of Canny edges across both Grayscale and HSV Saturation channels to capture dark-themed slides. Validates polygon contours (`approxPolyDP`, aspect ratio 1.2–2.2, extent > 0.85) to cleanly isolate shared content from webcam overlays, meeting tiles, and window borders.
- **Bounding Box Smoothing & Debounce**: Employs Exponential Moving Average (EMA) smoothing and a 3-frame state debounce mechanism to handle layout shifts without visual jitter or false cuts.
- **PTS-Based Sampling**: Samples frames strictly via Presentation Timestamps (PTS) using PyAV, guaranteeing frame accuracy on Variable Frame Rate (VFR) meeting recordings.
- **Perceptual Deduplication**: Filters near-duplicate slides using perceptual hashing with a configurable transition cooldown.
- **Integrated OCR**: Performs text extraction with mean confidence scores via Tesseract.

### 2. Deictic Physical & Natural Video Understanding
- **Syntactic Gesture & Reference Parsing**: Uses spaCy dependency parsing to detect spoken deictic references (*"this button"*, *"this popup"*, *"this board"*, *"that pin"*, *"these connectors"*) and spatial cues.
- **Optional BYOM Intent Verification**: Supports Bring-Your-Own-Model (BYOM) via any LangChain chat model to verify whether spoken statements genuinely reference visual elements, filtering out conversational and rhetorical remarks.
- **Kinematic & Temporal Alignment**: Maps spoken utterances to visual keyframe intervals based on exact ASR word timestamps.
- **Multi-Level Deduplication**: Suppresses redundant visual captures using temporal sliding windows and semantic overlap checks.

### 3. Unified Data Contracts & Storage Backends
- **Standardized Visual Segments**: Emits structured, modality-neutral [`VisualSegment`](#core-data-model-visualsegment) dataclasses with flat scalar metadata ready for vector store indexing.
- **Pluggable Storage**: Store extracted keyframes locally ([`LocalStorage`](#storage-backends)), in-memory as bytes ([`MemoryStorage`](#storage-backends)) or pipe them directly to cloud storage (S3, GCS, Azure Blob) via [`CallbackStorage`](#storage-backends).

---

## Installation

```bash
pip install visualrag
```

### System Requirements

- **Tesseract OCR (Optional)**: Needed only if extracting OCR text from presentation slides or screen shares. If not installed, `visualrag` runs without error, extracting slides, boundaries, and transition timestamps with `text=""`.
  - **Debian / Ubuntu**: `sudo apt install tesseract-ocr`
  - **macOS**: `brew install tesseract`
  - **Windows**:
    - Via Windows Package Manager: `winget install UB-Mannheim.Tesseract-OCR`
    - Or download the installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and ensure the install folder (e.g. `C:\Program Files\Tesseract-OCR`) is added to your system `PATH`.

*(Note: The spaCy `en_core_web_sm` model used for natural video syntactic parsing is downloaded automatically on first use. For containerized, read-only or offline environments, pre-install it via `python -m spacy download en_core_web_sm`).*

---

## Quickstart

### Basic Extraction Pipeline

```python
from visualrag import VisualRAG, LocalStorage
from visualrag.models import TranscriptSegment

# 1. Initialize pipeline with a storage backend
pipeline = VisualRAG(storage=LocalStorage(output_dir="./extracted_slides"))

# 2. Provide optional transcript segments with timestamps
transcript = [
    TranscriptSegment(start_sec=0.0, end_sec=6.5, text="Welcome everyone. Let's look at the database schema."),
    TranscriptSegment(start_sec=6.5, end_sec=14.0, text="Here you can see the primary relations and foreign keys."),
]

# 3. Process video (automatically classifies screen recording vs. natural video)
segments = pipeline.process(
    video="meeting_recording.mp4",
    transcript=transcript,
)

# 4. Access structured visual segments
for segment in segments:
    print(f"Segment ID : {segment.id}")
    print(f"Time Range : {segment.start_time_sec:.2f}s - {segment.end_time_sec:.2f}s")
    print(f"Image URI  : {segment.image_uri}")
    print(f"OCR Text   :\n{segment.text}")
    print(f"Transcript :\n{segment.transcript}")
    print("-" * 50)
```

---

## Core Data Model: `VisualSegment`

Every extraction pipeline returns a list of standardized `VisualSegment` objects:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique segment identifier (`slide_0001`, `deictic_0002`). |
| `modality` | `VideoModality` | `VideoModality.SCREEN_RECORDING` or `VideoModality.NATURAL_VIDEO`. |
| `source_video` | `str` | Filename or identifier of the source video. |
| `start_time_sec` | `float` | Timestamp when the visual state became active. |
| `end_time_sec` | `float` | Timestamp when the visual state ended or was replaced. |
| `duration_sec` | `float` | Display duration in seconds (`end_time_sec - start_time_sec`). |
| `image_uri` | `str \| None` | Storage locator for the saved keyframe (e.g. file path or S3 URI). |
| `image_bytes` | `bytes \| None` | Raw JPEG bytes (populated when using `MemoryStorage`). |
| `text` | `str` | Extracted OCR text for screen recordings or descriptive content. |
| `transcript` | `str \| None` | Spoken dialogue aligned to this visual segment. |
| `metadata` | `dict[str, Any]` | Modality-specific metadata (bounding box, OCR confidence, pHash, etc.). |

---

## Pipeline Initialization & Configurations

`VisualRAG` adapts cleanly to different execution environments, frame persistence strategies, and sampling granularities:

```python
from visualrag import VisualRAG, LocalStorage, MemoryStorage, CallbackStorage
from langchain_openai import ChatOpenAI

# 1. Default In-Memory Storage (Serverless, APIs, or Streaming)
# Keyframes are held as raw JPEG bytes in memory (available via segment.image_bytes)
pipeline = VisualRAG()

# 2. Local Filesystem Storage
# Writes frames to a local folder and populates segment.image_uri with file:// paths
pipeline = VisualRAG(output_dir="./extracted_slides")

# 3. Direct Cloud Object Storage (S3, GCS, Azure Blob)
def upload_to_s3(image_bytes: bytes, frame_id: str, metadata: dict) -> str:
    s3_key = f"recordings/{metadata.get('source_video', 'common')}/{frame_id}.jpg"
    s3_client.put_object(Bucket="my-rag-bucket", Key=s3_key, Body=image_bytes)
    return f"s3://my-rag-bucket/{s3_key}"

pipeline = VisualRAG(storage_backend=CallbackStorage(save_fn=upload_to_s3))

# 4. Fast-Paced Presentations (Custom PTS sampling and sensitivity)
pipeline = VisualRAG(
    output_dir="./slides",
    slide_sample_interval=0.5,    # Sample every 0.5s (default: 1.0s)
    slide_change_threshold=0.08,   # Sensitive cut detection for subtle slide transitions
)

# 5. Semantic LLM Verification for Natural Video Deictic Events
# Uses an LLM to verify whether spoken utterances genuinely reference visual physical items
pipeline = VisualRAG(
    output_dir="./frames",
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)
```

---

## VisualRAG Retrieval & Agent Pattern

In a typical VisualRAG architecture, spoken transcript segments are indexed into a vector database alongside their visual metadata (`segment_id`, `source_video`, `image_uri`, `modality`). 

A downstream LangChain RAG agent retrieves the relevant spoken dialogue passages, and uses a **`retrieve_visual_context`** tool on-demand to inspect the associated visual keyframe only when visual clarification is needed.

### 1. Ingesting Transcript Segments with Linked Visual Metadata

In this pattern, the complete meeting transcript is vectorized segment-by-segment into your vector database. For every transcript segment, if `VisualRAG` detected an active presentation slide or visual demonstration during that time range, the corresponding `image_uri` and visual metadata are attached directly to that transcript chunk:

```python
import boto3
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from visualrag import VisualRAG
from visualrag.models import TranscriptSegment

s3_client = boto3.client("s3")

# Simple S3 upload callback (no CallbackStorage wrapper needed)
def upload_to_s3(image_bytes: bytes, filename: str) -> str:
    s3_key = f"recordings/{filename}"
    s3_client.put_object(Bucket="my-rag-bucket", Key=s3_key, Body=image_bytes)
    return f"s3://my-rag-bucket/{s3_key}"

# 1. Initialize VisualRAG directly with the S3 callback and extract segments
pipeline = VisualRAG(storage_backend=upload_to_s3)
visual_segments = pipeline.process("meeting.mp4", transcript=full_transcript)

# 2. Iterate through the transcript using a two-pointer sweep (O(N + M))
texts = []
metadatas = []
vis_idx = 0
num_visuals = len(visual_segments)

for t_seg in full_transcript:
    texts.append(t_seg.text)

    # Advance pointer while current visual segment ended before this speech began
    while vis_idx < num_visuals and visual_segments[vis_idx].end_time_sec <= t_seg.start_sec:
        vis_idx += 1

    # Check if the active visual segment overlaps with this transcript segment
    matching_visual = None
    if vis_idx < num_visuals:
        v = visual_segments[vis_idx]
        if max(t_seg.start_sec, v.start_time_sec) < min(t_seg.end_sec, v.end_time_sec):
            matching_visual = v

    metadata = {
        "source_video": "meeting.mp4",
        "start_time_sec": t_seg.start_sec,
        "end_time_sec": t_seg.end_sec,
    }

    # Only attach visual pointers when an active slide/keyframe was present
    if matching_visual and matching_visual.image_uri:
        metadata["image_uri"] = matching_visual.image_uri
        metadata["visual_segment_id"] = matching_visual.id
        metadata["modality"] = matching_visual.modality.value

    metadatas.append(metadata)

# 3. Ingest all transcript segments into the vector database
vector_store = FAISS.from_texts(texts=texts, embedding=OpenAIEmbeddings(), metadatas=metadatas)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

### 2. Multi-Source RAG Agent with On-Demand Visual Retrieval

In a production knowledge base, your vector database often indexes multiple documentation sources (engineering wikis, design specs, and video transcripts). 

A standard enterprise RAG agent queries the knowledge base, and whenever a retrieved chunk includes an `image_uri` in its metadata and discusses visual elements, the agent calls `retrieve_visual_context` to inspect the visual keyframe:

```python
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool

# 1. Generic knowledge base retrieval tool (searches docs, transcripts, and notes)
retriever_tool = create_retriever_tool(
    retriever,
    name="search_knowledge_base",
    description=(
        "Searches the knowledge base across technical documentation, architecture notes, "
        "and meeting transcripts to find relevant information for user queries."
    ),
)

import base64

# 2. On-demand visual inspection tool (fetches frame from S3 and returns multimodal payload)
@tool
def retrieve_visual_context(image_uri: str) -> dict:
    """Fetches the visual context (presentation slide, diagram, UI screen,
    or video demonstration keyframe) from S3 and returns it for visual inspection.
    Call this tool whenever a retrieved document chunk contains an `image_uri` in its metadata
    and references visual information needed to provide a complete answer.
    """
    # Fetch the image bytes from S3
    bucket, key = image_uri.replace("s3://", "").split("/", 1)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()

    # Encode as Base64 data URL so multimodal models (e.g. GPT-4o) directly inspect the image
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
    }

# 3. Assemble the production agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [retriever_tool, retrieve_visual_context]

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI assistant helping engineers and analysts with technical queries. "
        "Search the knowledge base for relevant context. When a retrieved chunk contains visual "
        "metadata (`image_uri`) and mentions visual assets (such as diagrams, slides, charts, "
        "or physical components), call `retrieve_visual_context` using that image_uri to verify "
        "the visual details and provide a comprehensive answer."
    )),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Query that retrieves transcript text, detects image_uri, and inspects the visual frame
response = agent_executor.invoke({
    "input": "How does our authentication service communicate with the API gateway according to the architecture review?"
})
print(response["output"])
```

---

## License

This project is licensed under the [MIT License](LICENSE).
