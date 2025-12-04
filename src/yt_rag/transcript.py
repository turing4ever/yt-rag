"""Transcript extraction using youtube-transcript-api."""

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api.proxies import GenericProxyConfig

from .config import get_proxy_url
from .models import Segment, Transcript


class TranscriptError(Exception):
    """Transient error fetching transcript (retriable)."""

    pass


class TranscriptUnavailable(TranscriptError):
    """Transcript not available for this video (permanent)."""

    pass


@retry(
    retry=retry_if_exception_type(TranscriptError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def fetch_transcript(video_id: str, languages: list[str] | None = None) -> Transcript:
    """Fetch transcript for a video with automatic retry on transient errors.

    Args:
        video_id: YouTube video ID
        languages: Preferred languages in order (default: ["en"])

    Returns:
        Transcript with segments

    Raises:
        TranscriptUnavailable: If no transcript available (permanent, no retry)
        TranscriptError: For transient errors (retried up to 3 times)
    """
    languages = languages or ["en"]

    proxy_url = get_proxy_url()
    if proxy_url:
        api = YouTubeTranscriptApi(proxy_config=GenericProxyConfig(http_url=proxy_url))
    else:
        api = YouTubeTranscriptApi()

    try:
        transcript_data = api.fetch(video_id, languages=languages)

        segments = []
        for seq, item in enumerate(transcript_data):
            segment = Segment(
                video_id=video_id,
                seq=seq,
                start_time=item.start,
                duration=item.duration,
                text=item.text,
            )
            segments.append(segment)

        return Transcript(video_id=video_id, segments=segments)

    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
        # Permanent failures - don't retry
        raise TranscriptUnavailable(f"No transcript available: {e}") from e
    except TranscriptUnavailable:
        # Re-raise without wrapping
        raise
    except Exception as e:
        # Transient errors - will be retried by decorator
        raise TranscriptError(f"Error fetching transcript: {e}") from e


def get_transcript_text(video_id: str, languages: list[str] | None = None) -> str:
    """Fetch transcript and return as plain text."""
    transcript = fetch_transcript(video_id, languages)
    return transcript.full_text
