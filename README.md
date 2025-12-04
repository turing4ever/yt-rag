# yt-rag

Extract YouTube transcripts for RAG (Retrieval-Augmented Generation) pipelines.

## Features

- **Channel Discovery**: Add YouTube channels and automatically track all videos
- **Transcript Extraction**: Download transcripts using `youtube-transcript-api`
- **RAG Processing**: GPT-powered semantic sectioning and summarization
- **Vector Search**: FAISS-based semantic search across video content
- **Question Answering**: Ask questions and get answers with source citations
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

# Fetch transcripts
yt-rag fetch

# Process videos (sectionize + summarize with GPT)
yt-rag process

# Build vector index
yt-rag embed

# Ask questions!
yt-rag ask "What is tokenization?"
```

## Commands

### Data Collection

| Command | Description |
|---------|-------------|
| `yt-rag init` | Initialize the database |
| `yt-rag add <url>` | Add a channel or video |
| `yt-rag sync` | Discover new videos from tracked channels |
| `yt-rag fetch` | Download transcripts for pending videos |
| `yt-rag list channels` | List tracked channels |
| `yt-rag list videos` | List videos |
| `yt-rag status` | Show database statistics |

### RAG Processing

| Command | Description |
|---------|-------------|
| `yt-rag process` | Sectionize and summarize videos with GPT |
| `yt-rag embed` | Build FAISS vector index from sections |
| `yt-rag ask "<query>"` | Ask a question using RAG |

### Export

| Command | Description |
|---------|-------------|
| `yt-rag export -o <file>` | Export chunks for RAG |
| `yt-rag transcript <video_id>` | Export single video transcript |

## RAG Pipeline

### 1. Process Videos

Process fetched transcripts into semantic sections and summaries:

```bash
# Process all fetched videos
yt-rag process

# Process single video
yt-rag process VIDEO_ID

# Options
yt-rag process --limit 10           # Limit number of videos
yt-rag process --sectionize         # Only create sections
yt-rag process --summarize          # Only create summaries
yt-rag process --model gpt-4o       # Use different model
yt-rag process --force              # Re-process existing
```

### 2. Build Vector Index

Embed sections into FAISS for semantic search:

```bash
# Embed all sections
yt-rag embed

# Embed single video
yt-rag embed VIDEO_ID

# Options
yt-rag embed --rebuild              # Rebuild entire index
yt-rag embed --force                # Re-embed existing
yt-rag embed -m text-embedding-3-large  # Different model
```

### 3. Ask Questions

Query your video content with RAG:

```bash
# Ask a question
yt-rag ask "What is tokenization?"

# Filter by channel or video
yt-rag ask "How does attention work?" -c CHANNEL_ID
yt-rag ask "Explain the architecture" -v VIDEO_ID

# Options
yt-rag ask "question" -k 10         # Retrieve more sources (default: 5)
yt-rag ask "question" --no-answer   # Search only, skip answer generation
```

Example output:
```
Answer:
Tokenization is the process of converting text into smaller units called tokens...

Sources (3):

1. Let's build GPT: from scratch
   Section: Tokenization and Encoding
   Score: 87% | https://youtube.com/watch?v=kCc8FmEb1nY&t=1234s

2. Building makemore Part 5
   Section: Character-level vs Subword Tokenization
   Score: 82% | https://youtube.com/watch?v=t3YJ5hKiMQ0&t=567s

Latency: 1523ms | Tokens: 42 embed + 1205 chat
```

## Configuration

Data is stored in `~/.yt-rag/`:

```
~/.yt-rag/
├── db.sqlite          # Video and transcript database
├── .env               # API keys
├── config.toml        # User settings (optional)
└── faiss/
    ├── sections.index     # FAISS vector index
    └── sections_meta.jsonl  # Index metadata
```

### Environment Variables

Create `~/.yt-rag/.env`:

```bash
# Required for RAG features
OPENAI_API_KEY=sk-...

# Optional: proxy for transcript fetching
WEB_PROXY_URL=http://user:pass@proxy:port
```

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

## Transcript Format

The `transcript` command exports a text file with timestamps:

```
Title: Video Title
URL: https://www.youtube.com/watch?v=VIDEO_ID
Video ID: VIDEO_ID
------------------------------------------------------------

[00:00] First segment text...
[00:06] Second segment text...
[01:23] Later segment text...
```

## Requirements

- Python 3.11+
- OpenAI API key (for RAG features)

## License

MIT
