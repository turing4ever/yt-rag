# yt-rag

Extract YouTube transcripts for RAG (Retrieval-Augmented Generation) pipelines.

## Features

- **Channel Discovery**: Add YouTube channels and automatically track all videos
- **Transcript Extraction**: Download transcripts using `youtube-transcript-api`
- **SQLite Storage**: Persistent local database for videos and transcripts
- **RAG-Ready Export**: Chunked JSONL/JSON output with metadata for vector databases
- **CLI Interface**: Simple commands with progress indicators

## Installation

```bash
pip install yt-rag
```

Or with uv:
```bash
uv add yt-rag
```

## Quick Start

```bash
# Initialize database
yt-rag init

# Add a YouTube channel
yt-rag add https://www.youtube.com/@SomeChannel

# Or add a single video
yt-rag add https://www.youtube.com/watch?v=VIDEO_ID

# Sync to discover new videos
yt-rag sync

# Fetch transcripts
yt-rag fetch

# Export for RAG pipeline
yt-rag export -o transcripts.jsonl
```

## Commands

| Command | Description |
|---------|-------------|
| `yt-rag init` | Initialize the database |
| `yt-rag add <url>` | Add a channel or video |
| `yt-rag sync` | Discover new videos from tracked channels |
| `yt-rag fetch` | Download transcripts for pending videos |
| `yt-rag export -o <file>` | Export chunks for RAG |
| `yt-rag list channels` | List tracked channels |
| `yt-rag list videos` | List videos |
| `yt-rag status` | Show database statistics |

## Export Format

Each chunk in the JSONL output contains:

```json
{
  "chunk_id": "VIDEO_ID_chunk_0001",
  "video_id": "VIDEO_ID",
  "video_title": "Video Title",
  "channel_id": "CHANNEL_ID",
  "channel_name": "Channel Name",
  "url": "https://www.youtube.com/watch?v=VIDEO_ID&t=120",
  "start_time": 120.0,
  "end_time": 180.0,
  "text": "Transcript text for this chunk..."
}
```

## Configuration

Data is stored in `~/.yt-rag/`:
- `db.sqlite` - Video and transcript database
- `.env` - API keys (for future LLM integrations)
- `config.toml` - User settings

## Requirements

- Python 3.11+

## License

MIT
