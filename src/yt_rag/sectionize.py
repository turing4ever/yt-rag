"""GPT-based semantic sectioning of video transcripts."""

import re
from dataclasses import dataclass

from .config import DEFAULT_CHAT_MODEL
from .db import Database
from .models import Section, Segment
from .openai_client import simple_chat


@dataclass
class SectionizeResult:
    """Result of sectionizing a video transcript."""

    video_id: str
    sections: list[Section]
    tokens_used: int


SECTIONIZE_SYSTEM = """You are a transcript analyzer. Your task is to break a video transcript \
into logical semantic sections.

For each section:
1. Identify natural topic boundaries where the speaker transitions to a new subject
2. Give each section a clear, descriptive title (3-8 words)
3. Note the approximate start time of each section

Output format (one section per line):
[START_TIME] TITLE
content of the section...

[START_TIME] TITLE
content of the section...

Rules:
- START_TIME should be the timestamp in seconds (integer) from the transcript
- Each section should be 100-500 words ideally
- Preserve the original transcript text exactly
- Do not add commentary or change wording
- Section titles should be descriptive and specific to the content
- Create 3-15 sections depending on video length"""

SECTIONIZE_PROMPT = """Break this transcript into semantic sections. The video is titled: "{title}"

Transcript with timestamps:
{transcript}

Output each section with its start time and a descriptive title."""


def format_transcript_with_times(segments: list[Segment]) -> str:
    """Format segments with timestamps for the prompt."""
    lines = []
    for seg in segments:
        minutes = int(seg.start_time // 60)
        seconds = int(seg.start_time % 60)
        lines.append(f"[{minutes}:{seconds:02d}] {seg.text}")
    return "\n".join(lines)


def parse_sections_response(response: str, video_id: str, segments: list[Segment]) -> list[Section]:
    """Parse GPT response into Section objects."""
    sections = []

    # Pattern to match section headers: [TIME] TITLE or [TIME]TITLE
    # Time can be seconds (e.g., [120]) or mm:ss (e.g., [2:00])
    section_pattern = re.compile(r"\[(\d+(?::\d+)?)\]\s*(.+)")

    current_section = None
    current_content_lines = []

    for line in response.split("\n"):
        match = section_pattern.match(line.strip())
        if match:
            # Save previous section if exists
            if current_section:
                content = "\n".join(current_content_lines).strip()
                if content:
                    current_section["content"] = content
                    sections.append(current_section)

            # Parse time
            time_str = match.group(1)
            if ":" in time_str:
                parts = time_str.split(":")
                start_time = int(parts[0]) * 60 + int(parts[1])
            else:
                start_time = int(time_str)

            title = match.group(2).strip()

            current_section = {
                "start_time": float(start_time),
                "title": title,
            }
            current_content_lines = []
        elif current_section is not None:
            # Add to current section content
            current_content_lines.append(line)

    # Don't forget the last section
    if current_section:
        content = "\n".join(current_content_lines).strip()
        if content:
            current_section["content"] = content
            sections.append(current_section)

    # Convert to Section objects with proper IDs and end times
    result = []
    for i, sec in enumerate(sections):
        # Calculate end time (start of next section or end of video)
        if i + 1 < len(sections):
            end_time = sections[i + 1]["start_time"]
        elif segments:
            last_seg = segments[-1]
            end_time = last_seg.start_time + last_seg.duration
        else:
            end_time = sec["start_time"] + 60  # Default 1 minute

        section = Section(
            id=f"{video_id}_section_{i:04d}",
            video_id=video_id,
            seq=i,
            title=sec["title"],
            content=sec["content"],
            start_time=sec["start_time"],
            end_time=end_time,
            word_count=len(sec["content"].split()),
        )
        result.append(section)

    return result


def sectionize_video(
    video_id: str,
    db: Database | None = None,
    model: str = DEFAULT_CHAT_MODEL,
) -> SectionizeResult:
    """Break a video transcript into semantic sections using GPT.

    Args:
        video_id: Video ID to process
        db: Database instance (creates one if not provided)
        model: Chat model to use

    Returns:
        SectionizeResult with sections and token usage
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

    # Format transcript
    transcript = format_transcript_with_times(segments)

    # Build prompt
    prompt = SECTIONIZE_PROMPT.format(
        title=video.title,
        transcript=transcript,
    )

    # Call GPT
    result = simple_chat(
        prompt=prompt,
        system=SECTIONIZE_SYSTEM,
        model=model,
        temperature=0.3,
        max_tokens=4000,
    )

    # Parse response
    sections = parse_sections_response(result.content, video_id, segments)

    # If parsing failed, create a single section with full transcript
    if not sections:
        full_text = " ".join(s.text for s in segments)
        sections = [
            Section(
                id=f"{video_id}_section_0000",
                video_id=video_id,
                seq=0,
                title=video.title,
                content=full_text,
                start_time=0.0,
                end_time=segments[-1].start_time + segments[-1].duration if segments else 0.0,
                word_count=len(full_text.split()),
            )
        ]

    # Save to database
    db.add_sections(sections)

    return SectionizeResult(
        video_id=video_id,
        sections=sections,
        tokens_used=result.tokens_input + result.tokens_output,
    )


def sectionize_batch(
    video_ids: list[str],
    db: Database | None = None,
    model: str = DEFAULT_CHAT_MODEL,
    skip_existing: bool = True,
) -> list[SectionizeResult]:
    """Sectionize multiple videos.

    Args:
        video_ids: List of video IDs to process
        db: Database instance
        model: Chat model to use
        skip_existing: Skip videos that already have sections

    Returns:
        List of SectionizeResult for processed videos
    """
    if db is None:
        db = Database()
        db.init()

    results = []
    for video_id in video_ids:
        # Check if already has sections
        if skip_existing:
            existing = db.get_sections(video_id)
            if existing:
                continue

        try:
            result = sectionize_video(video_id, db, model)
            results.append(result)
        except Exception as e:
            # Log error but continue with other videos
            print(f"Error sectionizing {video_id}: {e}")

    return results
