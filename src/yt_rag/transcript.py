"""Transcript extraction using youtube-transcript-api."""

from requests.exceptions import ConnectionError, SSLError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
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
    """Error fetching transcript."""

    pass


class TranscriptUnavailable(TranscriptError):
    """Transcript not available for this video."""

    pass


@retry(
    retry=retry_if_exception_type((SSLError, ConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    reraise=True,
)
def _fetch(api: YouTubeTranscriptApi, video_id: str, languages: list[str]):
    """Fetch transcript data from YouTube."""
    return api.fetch(video_id, languages=languages)


def fetch_transcript(video_id: str, languages: list[str] | None = None) -> Transcript:
    """Fetch transcript for a video.

    Args:
        video_id: YouTube video ID
        languages: Preferred languages in order (default: ["en"])

    Returns:
        Transcript with segments

    Raises:
        TranscriptUnavailable: If no transcript available
        TranscriptError: For other errors
    """
    languages = languages or ["en"]

    proxy_url = get_proxy_url()
    if proxy_url:
        api = YouTubeTranscriptApi(proxy_config=GenericProxyConfig(http_url=proxy_url))
    else:
        api = YouTubeTranscriptApi()

    try:
        transcript_data = _fetch(api, video_id, languages)

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
        raise TranscriptUnavailable(f"No transcript available: {e}")
    except Exception as e:
        raise TranscriptError(f"Error fetching transcript: {e}")


def get_transcript_text(video_id: str, languages: list[str] | None = None) -> str:
    """Fetch transcript and return as plain text."""
    transcript = fetch_transcript(video_id, languages)
    return transcript.full_text
