"""Pydantic models for yt-rag."""

from datetime import datetime

from pydantic import BaseModel


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
    transcript_status: str = "pending"  # pending, fetched, unavailable
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
