"""YouTube channel/video discovery using yt-dlp."""

import re
from datetime import datetime

import yt_dlp

from .models import Channel, Video


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


def get_channel_info(url: str) -> Channel:
    """Fetch channel metadata from YouTube."""
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

    return Channel(
        id=channel_id,
        name=channel_name,
        url=channel_url,
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
    """Get info for a single video."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id = info.get("id")
    return Video(
        id=video_id,
        channel_id=info.get("channel_id"),
        title=info.get("title", "Unknown"),
        url=f"https://www.youtube.com/watch?v={video_id}",
        published_at=_parse_upload_date(info.get("upload_date")),
        duration_seconds=info.get("duration"),
        transcript_status="pending",
    )
