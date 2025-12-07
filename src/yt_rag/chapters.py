"""Chapter-based sectionizing of video transcripts.

Uses YouTube chapters when available, falls back to time-based chunking.
Optionally generates GPT titles for time-based chunks.

Chapter duration is calculated based on analysis of human-labeled YouTube chapters:
- Short videos (<10min): ~1.7 min chapters, ~6 per 10min
- Medium videos (10-30min): ~3.1 min chapters, ~3.5 per 10min
- Long videos (30-60min): ~4.0 min chapters, ~2.5 per 10min
- Very long (60+min): ~4.5 min chapters, ~2.3 per 10min
"""

from dataclasses import dataclass

from .config import DEFAULT_CHAT_MODEL, DEFAULT_CHUNK_DURATION
from .db import Database
from .discovery import get_video_chapters
from .models import Chapter, Section, Segment
from .openai_client import simple_chat


def get_optimal_chunk_duration(video_duration_seconds: float | None) -> float:
    """Calculate optimal chunk duration based on video length.

    Based on analysis of 1000+ human-labeled YouTube chapters:
    - Shorter videos have shorter, more frequent chapters
    - Longer videos have longer chapters

    Args:
        video_duration_seconds: Video duration in seconds

    Returns:
        Optimal chunk duration in seconds
    """
    if video_duration_seconds is None:
        return DEFAULT_CHUNK_DURATION

    duration_min = video_duration_seconds / 60

    # Empirical data from YouTube chapter analysis:
    # Duration bucket -> avg chapter length (minutes)
    if duration_min < 10:
        chunk_min = 1.7
    elif duration_min < 20:
        chunk_min = 3.1
    elif duration_min < 30:
        chunk_min = 3.8
    elif duration_min < 45:
        chunk_min = 3.6
    elif duration_min < 60:
        chunk_min = 4.9
    else:
        chunk_min = 4.5

    return chunk_min * 60  # Convert to seconds


@dataclass
class SectionizeResult:
    """Result of sectionizing a video transcript."""

    video_id: str
    sections: list[Section]
    method: str  # "chapters" or "time_chunks"


def _get_segment_text_in_range(
    segments: list[Segment], start_time: float, end_time: float
) -> str:
    """Get concatenated text from segments within a time range."""
    texts = []
    for seg in segments:
        seg_end = seg.start_time + seg.duration
        # Include segment if it overlaps with the range
        if seg.start_time < end_time and seg_end > start_time:
            texts.append(seg.text)
    return " ".join(texts)


def sectionize_by_chapters(
    video_id: str,
    segments: list[Segment],
    chapters: list[Chapter],
    video_duration: float | None = None,
) -> list[Section]:
    """Create sections from YouTube chapters.

    Args:
        video_id: Video ID
        segments: Transcript segments
        chapters: YouTube chapters
        video_duration: Total video duration in seconds

    Returns:
        List of Section objects
    """
    if not chapters:
        return []

    sections = []
    for i, chapter in enumerate(chapters):
        # Determine end time
        if chapter.end_time:
            end_time = chapter.end_time
        elif video_duration:
            end_time = video_duration
        elif segments:
            last_seg = segments[-1]
            end_time = last_seg.start_time + last_seg.duration
        else:
            end_time = chapter.start_time + 300  # Default 5 min

        content = _get_segment_text_in_range(segments, chapter.start_time, end_time)

        section = Section(
            id=f"{video_id}_section_{i:04d}",
            video_id=video_id,
            seq=i,
            title=chapter.title,
            content=content,
            start_time=chapter.start_time,
            end_time=end_time,
            word_count=len(content.split()) if content else 0,
        )
        sections.append(section)

    return sections


TITLE_PROMPT = """Generate a short, descriptive title (3-8 words) for this video transcript section.
The title should capture the main topic being discussed.

Video: {video_title}
Timestamp: {timestamp}

Transcript:
{content}

Reply with ONLY the title, nothing else."""


def _generate_section_title(
    content: str,
    video_title: str,
    timestamp: str,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """Generate a title for a section using GPT."""
    # Truncate content to ~500 words to keep costs low
    words = content.split()
    if len(words) > 500:
        content = " ".join(words[:500]) + "..."

    prompt = TITLE_PROMPT.format(
        video_title=video_title,
        timestamp=timestamp,
        content=content,
    )

    result = simple_chat(prompt, model=model, temperature=0.3, max_tokens=50)
    return result.content.strip().strip('"')


def sectionize_by_time(
    video_id: str,
    segments: list[Segment],
    video_title: str,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    generate_titles: bool = False,
    model: str = DEFAULT_CHAT_MODEL,
) -> list[Section]:
    """Create sections by splitting transcript into time-based chunks.

    Args:
        video_id: Video ID
        segments: Transcript segments
        video_title: Video title (used for naming)
        chunk_duration: Duration of each chunk in seconds
        generate_titles: Whether to generate GPT titles for chunks
        model: Model to use for title generation

    Returns:
        List of Section objects
    """
    if not segments:
        return []

    # Get total duration
    last_seg = segments[-1]
    total_duration = last_seg.start_time + last_seg.duration

    sections = []
    chunk_start = 0.0
    chunk_idx = 0

    while chunk_start < total_duration:
        chunk_end = min(chunk_start + chunk_duration, total_duration)
        content = _get_segment_text_in_range(segments, chunk_start, chunk_end)

        if content.strip():
            # Create title
            minutes = int(chunk_start // 60)
            timestamp = f"{minutes}:{int(chunk_start % 60):02d}"

            if generate_titles:
                title = _generate_section_title(content, video_title, timestamp, model)
            else:
                title = f"Part {chunk_idx + 1} ({timestamp})"

            section = Section(
                id=f"{video_id}_section_{chunk_idx:04d}",
                video_id=video_id,
                seq=chunk_idx,
                title=title,
                content=content,
                start_time=chunk_start,
                end_time=chunk_end,
                word_count=len(content.split()),
            )
            sections.append(section)
            chunk_idx += 1

        chunk_start = chunk_end

    return sections


def sectionize_video(
    video_id: str,
    db: Database | None = None,
    chunk_duration: float | None = None,
    generate_titles: bool = False,
    model: str = DEFAULT_CHAT_MODEL,
) -> SectionizeResult:
    """Sectionize a video transcript using chapters or time-based fallback.

    Priority:
    1. YouTube chapters (if available)
    2. Time-based chunks with optional GPT titles (fallback)

    Args:
        video_id: Video ID to process
        db: Database instance (creates one if not provided)
        chunk_duration: Duration for time-based chunks (seconds).
            If None, uses data-driven optimal duration based on video length.
        generate_titles: Generate GPT titles for time-based chunks
        model: Model for title generation

    Returns:
        SectionizeResult with sections and method used
    """
    if db is None:
        db = Database()
        db.init()

    # Get video and segments
    video = db.get_video(video_id)
    if not video:
        raise ValueError(f"Video not found: {video_id}")

    segments = db.get_segments(video_id)
    if not segments:
        raise ValueError(f"No segments found for video: {video_id}")

    # Try YouTube chapters first
    chapters = get_video_chapters(video_id)

    if chapters:
        sections = sectionize_by_chapters(
            video_id=video_id,
            segments=segments,
            chapters=chapters,
            video_duration=float(video.duration_seconds) if video.duration_seconds else None,
        )
        method = "chapters"
    else:
        # Fall back to time-based chunks with data-driven duration
        if chunk_duration is None:
            chunk_duration = get_optimal_chunk_duration(video.duration_seconds)

        sections = sectionize_by_time(
            video_id=video_id,
            segments=segments,
            video_title=video.title,
            chunk_duration=chunk_duration,
            generate_titles=generate_titles,
            model=model,
        )
        method = "time_chunks_gpt" if generate_titles else "time_chunks"

    # Save to database
    if sections:
        db.add_sections(sections)

    return SectionizeResult(
        video_id=video_id,
        sections=sections,
        method=method,
    )


def sectionize_batch(
    video_ids: list[str],
    db: Database | None = None,
    chunk_duration: float | None = None,
    skip_existing: bool = True,
    generate_titles: bool = False,
    model: str = DEFAULT_CHAT_MODEL,
) -> list[SectionizeResult]:
    """Sectionize multiple videos.

    Args:
        video_ids: List of video IDs to process
        db: Database instance
        chunk_duration: Duration for time-based chunks (None = auto based on video length)
        skip_existing: Skip videos that already have sections
        generate_titles: Generate GPT titles for time-based chunks
        model: Model for title generation

    Returns:
        List of SectionizeResult for processed videos
    """
    if db is None:
        db = Database()
        db.init()

    results = []
    for video_id in video_ids:
        if skip_existing:
            existing = db.get_sections(video_id)
            if existing:
                continue

        try:
            result = sectionize_video(
                video_id, db, chunk_duration, generate_titles=generate_titles, model=model
            )
            results.append(result)
        except Exception as e:
            print(f"Error sectionizing {video_id}: {e}")

    return results
