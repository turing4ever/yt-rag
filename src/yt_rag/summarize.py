"""GPT-based video summarization."""

from dataclasses import dataclass

from .config import DEFAULT_CHAT_MODEL
from .db import Database
from .models import Section, Summary
from .openai_client import simple_chat


@dataclass
class SummarizeResult:
    """Result of summarizing a video."""

    video_id: str
    summary: Summary
    tokens_used: int


SUMMARIZE_SYSTEM = """You are a video content summarizer. Create clear, informative summaries \
that capture the key points and value of video content.

Your summaries should:
1. Start with a 1-2 sentence overview of what the video covers
2. List 3-7 key points or takeaways
3. Note any important conclusions or recommendations
4. Be factual and based only on the provided content

Format:
## Overview
[1-2 sentence summary]

## Key Points
- [point 1]
- [point 2]
...

## Takeaways
[Notable conclusions or actionable insights]"""


SUMMARIZE_FROM_SECTIONS_PROMPT = """Summarize this video based on its sections.

Video Title: {title}

Sections:
{sections}

Create a comprehensive summary."""


SUMMARIZE_FROM_TRANSCRIPT_PROMPT = """Summarize this video based on its transcript.

Video Title: {title}

Transcript:
{transcript}

Create a comprehensive summary."""


def format_sections_for_summary(sections: list[Section]) -> str:
    """Format sections for the summarization prompt."""
    parts = []
    for sec in sections:
        parts.append(f"### {sec.title}")
        # Truncate very long sections
        content = sec.content
        if len(content) > 2000:
            content = content[:2000] + "..."
        parts.append(content)
        parts.append("")
    return "\n".join(parts)


def summarize_video(
    video_id: str,
    db: Database | None = None,
    model: str = DEFAULT_CHAT_MODEL,
    use_sections: bool = True,
) -> SummarizeResult:
    """Generate a summary for a video using GPT.

    Args:
        video_id: Video ID to summarize
        db: Database instance (creates one if not provided)
        model: Chat model to use
        use_sections: Use sections if available, otherwise use raw transcript

    Returns:
        SummarizeResult with summary and token usage
    """
    if db is None:
        db = Database()
        db.init()

    # Get video
    video = db.get_video(video_id)
    if not video:
        raise ValueError(f"Video not found: {video_id}")

    # Try to use sections first
    sections = db.get_sections(video_id) if use_sections else []

    if sections:
        # Use sections for better context
        sections_text = format_sections_for_summary(sections)
        prompt = SUMMARIZE_FROM_SECTIONS_PROMPT.format(
            title=video.title,
            sections=sections_text,
        )
    else:
        # Fall back to raw transcript
        transcript = db.get_full_text(video_id)
        if not transcript:
            raise ValueError(f"No transcript found for video: {video_id}")

        # Truncate if too long
        words = transcript.split()
        if len(words) > 6000:
            transcript = " ".join(words[:6000]) + "..."

        prompt = SUMMARIZE_FROM_TRANSCRIPT_PROMPT.format(
            title=video.title,
            transcript=transcript,
        )

    # Call GPT
    result = simple_chat(
        prompt=prompt,
        system=SUMMARIZE_SYSTEM,
        model=model,
        temperature=0.3,
        max_tokens=1500,
    )

    # Create summary object
    summary = Summary(
        video_id=video_id,
        summary=result.content,
    )

    # Save to database
    db.add_summary(summary)

    return SummarizeResult(
        video_id=video_id,
        summary=summary,
        tokens_used=result.tokens_input + result.tokens_output,
    )


def summarize_batch(
    video_ids: list[str],
    db: Database | None = None,
    model: str = DEFAULT_CHAT_MODEL,
    skip_existing: bool = True,
    use_sections: bool = True,
) -> list[SummarizeResult]:
    """Summarize multiple videos.

    Args:
        video_ids: List of video IDs to process
        db: Database instance
        model: Chat model to use
        skip_existing: Skip videos that already have summaries
        use_sections: Use sections if available

    Returns:
        List of SummarizeResult for processed videos
    """
    if db is None:
        db = Database()
        db.init()

    results = []
    for video_id in video_ids:
        # Check if already has summary
        if skip_existing:
            existing = db.get_summary(video_id)
            if existing:
                continue

        try:
            result = summarize_video(video_id, db, model, use_sections)
            results.append(result)
        except Exception as e:
            # Log error but continue with other videos
            print(f"Error summarizing {video_id}: {e}")

    return results
