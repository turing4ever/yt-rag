"""YouTube channel/video discovery using yt-dlp."""

import re
from datetime import datetime

import yt_dlp

from .models import Channel, Chapter, Video


def _parse_upload_date(upload_date: str | None) -> datetime | None:
    """Parse yt-dlp upload date string (YYYYMMDD) to datetime."""
    if not upload_date:
        return None
    try:
        return datetime.strptime(upload_date, "%Y%m%d")
    except ValueError:
        return None


def extract_channel_id(url: str) -> str | None:
    """Extract channel ID from various YouTube URL formats."""
    patterns = [
        r"youtube\.com/channel/([a-zA-Z0-9_-]+)",
        r"youtube\.com/@([a-zA-Z0-9_-]+)",
        r"youtube\.com/c/([a-zA-Z0-9_-]+)",
        r"youtube\.com/user/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def extract_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_channel_info(url: str, fetch_metadata: bool = True) -> Channel:
    """Fetch channel metadata from YouTube.

    Args:
        url: Channel URL
        fetch_metadata: If True, fetches full metadata (description, subs, tags).
                       If False, only fetches basic info (faster for listing videos).
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlist_items": "0",  # Don't fetch any videos, just channel info
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    channel_id = info.get("channel_id") or info.get("uploader_id") or extract_channel_id(url)
    channel_name = info.get("channel") or info.get("uploader") or "Unknown"
    channel_url = info.get("channel_url") or url

    # Extract additional metadata
    description = info.get("description")
    subscriber_count = info.get("channel_follower_count")
    tags = info.get("tags") or []
    handle = info.get("uploader_id")  # @handle

    return Channel(
        id=channel_id,
        name=channel_name,
        url=channel_url,
        description=description,
        subscriber_count=subscriber_count,
        tags=tags if tags else None,
        handle=handle,
    )


def list_channel_videos(url: str, channel_id: str | None = None) -> list[Video]:
    """List all videos from a YouTube channel."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,  # Don't download, just list
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    videos = []
    entries = info.get("entries", [])

    if not entries and info.get("id"):
        entries = [info]

    for entry in entries:
        if entry is None:
            continue

        video_id = entry.get("id")
        if not video_id:
            continue

        video = Video(
            id=video_id,
            channel_id=channel_id or entry.get("channel_id"),
            title=entry.get("title", "Unknown"),
            url=f"https://www.youtube.com/watch?v={video_id}",
            published_at=_parse_upload_date(entry.get("upload_date")),
            duration_seconds=entry.get("duration"),
            transcript_status="pending",
        )
        videos.append(video)

    return videos


def get_video_info(url: str) -> Video:
    """Get info for a single video including description and tags."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id = info.get("id")

    # Get tags and categories - yt-dlp returns as lists
    tags = info.get("tags") or []
    categories = info.get("categories") or []

    # Host defaults to channel/uploader name
    host = info.get("channel") or info.get("uploader")

    return Video(
        id=video_id,
        channel_id=info.get("channel_id"),
        title=info.get("title", "Unknown"),
        url=f"https://www.youtube.com/watch?v={video_id}",
        published_at=_parse_upload_date(info.get("upload_date")),
        duration_seconds=info.get("duration"),
        view_count=info.get("view_count"),
        like_count=info.get("like_count"),
        comment_count=info.get("comment_count"),
        description=info.get("description"),
        tags=tags if tags else None,
        categories=categories if categories else None,
        language=info.get("language"),
        host=host,
        availability=info.get("availability"),
        transcript_status="pending",
    )


def get_video_chapters(video_id: str) -> list[Chapter]:
    """Get chapters for a video from YouTube.

    Returns empty list if no chapters available.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    chapters_data = info.get("chapters") or []
    if not chapters_data:
        return []

    duration = info.get("duration") or 0

    chapters = []
    for i, ch in enumerate(chapters_data):
        start = ch.get("start_time", 0)
        # End time is start of next chapter, or video duration for last chapter
        if i + 1 < len(chapters_data):
            end = chapters_data[i + 1].get("start_time", start)
        else:
            end = duration if duration else None

        chapters.append(
            Chapter(
                title=ch.get("title", f"Chapter {i + 1}"),
                start_time=float(start),
                end_time=float(end) if end else None,
            )
        )

    return chapters
