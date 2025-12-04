"""Pydantic models for yt-rag."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

TranscriptStatus = Literal["pending", "fetched", "unavailable", "error"]


class Channel(BaseModel):
    """YouTube channel metadata."""

    id: str
    name: str
    url: str
    last_synced_at: datetime | None = None


class Video(BaseModel):
    """YouTube video metadata."""

    id: str
    channel_id: str | None = None
    title: str
    url: str
    published_at: datetime | None = None
    duration_seconds: int | None = None
    transcript_status: TranscriptStatus = "pending"
    created_at: datetime | None = None


class Segment(BaseModel):
    """Transcript segment with timing."""

    video_id: str
    seq: int
    start_time: float
    duration: float
    text: str


class Transcript(BaseModel):
    """Full transcript for a video."""

    video_id: str
    segments: list[Segment]

    @property
    def full_text(self) -> str:
        """Combine all segments into full text."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


class Chunk(BaseModel):
    """Chunked text for RAG export."""

    chunk_id: str
    video_id: str
    video_title: str
    channel_id: str | None = None
    channel_name: str | None = None
    url: str  # YouTube URL with timestamp
    start_time: float
    end_time: float
    text: str


ProcessStatus = Literal["pending", "sectioned", "summarized", "embedded", "error"]


class Section(BaseModel):
    """Semantic section of a video transcript."""

    id: str  # {video_id}_section_{seq:04d}
    video_id: str
    seq: int
    title: str
    content: str
    start_time: float | None = None
    end_time: float | None = None
    word_count: int | None = None
    created_at: datetime | None = None


class Summary(BaseModel):
    """AI-generated video summary."""

    video_id: str
    summary: str
    created_at: datetime | None = None


class QueryLog(BaseModel):
    """Log entry for a RAG query."""

    id: str
    query: str
    scope_type: str | None = None  # 'channel', 'video', or None
    scope_id: str | None = None
    retrieved_ids: list[str] | None = None
    retrieved_scores: list[float] | None = None
    answer: str | None = None
    latency_ms: int | None = None
    tokens_embedding: int | None = None
    tokens_chat: int | None = None
    model_embedding: str | None = None
    model_chat: str | None = None
    created_at: datetime | None = None


class Feedback(BaseModel):
    """User feedback on a query."""

    query_id: str
    helpful: bool | None = None
    source_rating: int | None = None  # 1-5
    comment: str | None = None
    created_at: datetime | None = None


class TestCase(BaseModel):
    """Benchmark test case."""

    id: str
    query: str
    scope_type: str | None = None
    scope_id: str | None = None
    expected_video_ids: list[str] | None = None
    expected_keywords: list[str] | None = None
    reference_answer: str | None = None
    created_at: datetime | None = None


class SearchResult(BaseModel):
    """A single search result from FAISS."""

    section_id: str
    video_id: str
    channel_id: str | None = None
    score: float
    section: Section | None = None
