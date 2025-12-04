"""Export transcripts for RAG pipelines."""

import json
from pathlib import Path

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .db import Database
from .models import Chunk, Segment


def build_word_to_time_map(segments: list[Segment]) -> list[tuple[float, float]]:
    """Build a mapping from word index to (start_time, end_time) for each word.

    Returns:
        List where index i contains (start_time, end_time) for word i.
        Time is interpolated within each segment based on word position.
    """
    word_times = []
    for seg in segments:
        words = seg.text.split()
        if not words:
            continue
        # Distribute segment duration evenly across its words
        time_per_word = seg.duration / len(words) if words else 0
        for i, _ in enumerate(words):
            word_start = seg.start_time + i * time_per_word
            word_end = seg.start_time + (i + 1) * time_per_word
            word_times.append((word_start, word_end))
    return word_times


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[int, int, str]]:
    """Split text into overlapping chunks by word count.

    Returns:
        List of (start_word_idx, end_word_idx, chunk_text)
    """
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [(0, len(words), text)]

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append((start, end, chunk_text))

        if end >= len(words):
            break
        start = end - overlap

    return chunks


def export_video_chunks(
    db: Database,
    video_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Export chunks for a single video."""
    video = db.get_video(video_id)
    if not video:
        return []

    segments = db.get_segments(video_id)
    if not segments:
        return []

    # Get channel info
    channel = db.get_channel(video.channel_id) if video.channel_id else None

    # Build full text and word-to-time mapping
    full_text = " ".join(s.text for s in segments)
    word_times = build_word_to_time_map(segments)

    # Chunk the text
    text_chunks = chunk_text(full_text, chunk_size, overlap)

    chunks = []
    for idx, (start_word, end_word, text) in enumerate(text_chunks):
        # Get actual times from the word mapping
        start_time = word_times[start_word][0] if start_word < len(word_times) else 0.0
        end_time = word_times[end_word - 1][1] if end_word <= len(word_times) else word_times[-1][1]

        chunk = Chunk(
            chunk_id=f"{video_id}_chunk_{idx:04d}",
            video_id=video_id,
            video_title=video.title,
            channel_id=video.channel_id,
            channel_name=channel.name if channel else None,
            url=f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}",
            start_time=start_time,
            end_time=end_time,
            text=text,
        )
        chunks.append(chunk)

    return chunks


def export_all_chunks(
    db: Database,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    channel_id: str | None = None,
) -> list[Chunk]:
    """Export chunks for all fetched videos."""
    videos = db.list_videos(channel_id=channel_id, status="fetched")
    all_chunks = []

    for video in videos:
        chunks = export_video_chunks(db, video.id, chunk_size, overlap)
        all_chunks.extend(chunks)

    return all_chunks


def export_to_jsonl(
    chunks: list[Chunk],
    output_path: Path,
) -> int:
    """Export chunks to JSONL file.

    Returns:
        Number of chunks written
    """
    with open(output_path, "w") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    return len(chunks)


def export_to_json(
    chunks: list[Chunk],
    output_path: Path,
) -> int:
    """Export chunks to JSON file.

    Returns:
        Number of chunks written
    """
    data = [chunk.model_dump() for chunk in chunks]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return len(chunks)
