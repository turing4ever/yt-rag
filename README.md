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

# Run full pipeline (sync, metadata, transcripts, process, embed)
yt-rag update

# Ask questions!
yt-rag ask "What is tokenization?"

# Or use interactive chat
yt-rag chat
```

## Commands

### Core Commands

| Command | Description |
|---------|-------------|
| `yt-rag init` | Initialize the database |
| `yt-rag add <url>` | Add a channel or video |
| `yt-rag update` | Run full pipeline (sync, metadata, transcripts, process, embed) |
| `yt-rag ask "<query>"` | Ask a question using RAG |
| `yt-rag chat` | Interactive chat with persistent sessions |
| `yt-rag status` | Show database statistics |

### Individual Pipeline Steps

| Command | Description |
|---------|-------------|
| `yt-rag sync-channel` | Discover new videos from tracked channels |
| `yt-rag refresh-meta` | Refresh video/channel metadata from YouTube |
| `yt-rag fetch-transcript` | Download transcripts for pending videos |
| `yt-rag process-transcript` | Sectionize and summarize videos |
| `yt-rag embed` | Build FAISS vector index from sections |

### Browsing

| Command | Description |
|---------|-------------|
| `yt-rag list channels` | List tracked channels |
| `yt-rag list videos` | List videos |
| `yt-rag videos` | Search/filter videos by metadata |

### Evaluation & Benchmarking

| Command | Description |
|---------|-------------|
| `yt-rag logs` | View query logs |
| `yt-rag feedback <id>` | Add feedback for a query |
| `yt-rag eval` | Run evaluation on manual test cases |
| `yt-rag test-add "<query>"` | Add a manual test case |
| `yt-rag test-list` | List all manual test cases |
| `yt-rag test-generate` | Generate benchmark tests from video content |
| `yt-rag test` | Run full RAG benchmark (classification, retrieval, answer quality) |
| `yt-rag test-report` | Generate HTML report for benchmark results |

### Search Enhancement

| Command | Description |
|---------|-------------|
| `yt-rag keywords` | Extract and analyze keywords from transcripts |
| `yt-rag synonyms` | Manage synonym mappings for search boosting |

### Export

| Command | Description |
|---------|-------------|
| `yt-rag export -o <file>` | Export chunks for RAG |
| `yt-rag transcript <video_id>` | Export single video transcript |

## Update Pipeline

The `update` command runs the complete pipeline automatically:

```bash
# Run full pipeline
yt-rag update

# Options
yt-rag update --test               # Test mode: 5 videos per channel
yt-rag update --skip-sync          # Skip channel sync
yt-rag update --skip-embed         # Skip embedding step
yt-rag update --force-transcript   # Re-fetch ALL transcripts
yt-rag update --force-meta         # Force refresh all metadata
yt-rag update --force-embed        # Rebuild all embeddings
yt-rag update --openai             # Use OpenAI for embeddings
```

Pipeline steps:
1. **Sync channels**: Pull new videos from tracked channels
2. **Refresh metadata**: Update video info (skips if refreshed within 1 day)
3. **Fetch transcripts**: Download transcripts for pending videos
4. **Process transcripts**: Sectionize (using YouTube chapters) and summarize
5. **Embed**: Build/update FAISS vector index

By default, each step only processes items that need work. Use `--force-*` flags to reprocess everything.

## RAG Search

### Ask Questions

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

### Interactive Chat

For multi-turn conversations with persistent sessions:

```bash
# Start or resume chat
yt-rag chat

# Use OpenAI instead of local Ollama
yt-rag chat --openai

# Filter to specific channel/video
yt-rag chat -c CHANNEL_ID
yt-rag chat -v VIDEO_ID

# Session management
yt-rag chat --new             # Start fresh session
yt-rag chat --list            # List recent sessions
yt-rag chat --session ID      # Resume specific session
```

In-chat commands:
- `/new` - Start a new session
- `/sessions` - List sessions
- `/rename <title>` - Rename current session

## Evaluation & Logging

### Query Logs

Every RAG query is automatically logged for analysis:

```bash
# View recent queries
yt-rag logs

# View more logs
yt-rag logs -n 50

# View specific query details
yt-rag logs -q QUERY_ID
```

### Feedback

Rate query results to track RAG quality:

```bash
# Mark a query as helpful/not helpful
yt-rag feedback QUERY_ID --helpful
yt-rag feedback QUERY_ID --not-helpful

# Rate source quality (1-5)
yt-rag feedback QUERY_ID -r 4

# Add a comment
yt-rag feedback QUERY_ID -c "Answer was accurate but missed key details"

# Combine options
yt-rag feedback QUERY_ID --helpful -r 5 -c "Great answer!"
```

### Benchmark Testing

Create test cases and run benchmarks to measure RAG quality:

```bash
# Add a test case
yt-rag test-add "What is tokenization?" --videos VIDEO_ID1,VIDEO_ID2
yt-rag test-add "Explain attention" --keywords "attention,transformer,query,key,value"
yt-rag test-add "How does backprop work?" --videos VIDEO_ID --keywords "gradient,chain rule"

# List test cases
yt-rag test-list

# Run benchmark
yt-rag eval

# Show failed tests only
yt-rag eval --failures

# Verbose output with per-test details
yt-rag eval -v
```

Metrics calculated:
- **Precision@K**: Fraction of top-K results that are relevant
- **Recall**: Fraction of relevant items found
- **MRR**: Mean Reciprocal Rank (how high is the first relevant result)
- **Keyword Match**: Fraction of expected keywords in the answer

## Configuration

Data is stored in `~/.yt-rag/`:

```
~/.yt-rag/
├── db.sqlite              # Video and transcript database
├── .env                   # API keys and settings
├── config.toml            # User settings (optional)
├── chat_history           # Readline history for chat
├── faiss/                 # OpenAI embeddings (1536 dims)
│   ├── sections.index
│   └── sections_meta.jsonl
└── faiss_local/           # Ollama embeddings (1024 dims, mxbai-embed-large)
    ├── sections.index
    └── sections_meta.jsonl
```

### Environment Variables

Create `~/.yt-rag/.env`:

```bash
# Required for --openai mode
OPENAI_API_KEY=sk-...

# Optional: proxy for transcript fetching
WEB_PROXY_URL=http://user:pass@proxy:port

# Optional: rate limiting for yt-dlp metadata fetching
YT_DLP_MAX_DELAY=0.5       # Max delay between requests (default: 0.5s)
YT_DLP_BATCH_SIZE=20       # Concurrent requests per batch (default: 20)
```

### Embedding Backends

By default, yt-rag uses local Ollama for embeddings and chat. Use `--openai` to switch to OpenAI.

**Local (default)**: Requires [Ollama](https://ollama.ai/) running locally.
```bash
# Start Ollama
sudo systemctl start ollama

# Pull required models
ollama pull mxbai-embed-large    # Embeddings (1024 dims)
ollama pull qwen2.5:7b-instruct  # Chat/answer generation
```

**OpenAI**: Requires `OPENAI_API_KEY` in `.env`.
```bash
yt-rag update --openai
yt-rag ask "question" --openai
yt-rag chat --openai
```

Local and OpenAI indexes are stored separately, so you can switch between them.

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

- Python 3.10+
- Ollama (default) or OpenAI API key (with `--openai`)

## Development

```bash
# Install dev dependencies
uv sync

# Run linter (must pass before push)
uv run ruff check src/

# Run formatter (must pass before push)
uv run ruff format --check src/

# Auto-fix formatting
uv run ruff format src/

# Run both checks (same as CI)
uv run ruff check src/ && uv run ruff format --check src/
```

## License

MIT
