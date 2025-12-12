# yt-rag

Extract YouTube transcripts for RAG (Retrieval-Augmented Generation) pipelines.

## Features

- **Channel Discovery**: Add YouTube channels and automatically track all videos
- **Transcript Extraction**: Download transcripts using `youtube-transcript-api`
- **RAG Processing**: LLM-powered semantic sectioning and summarization
- **Vector Search**: FAISS-based semantic search across video content
- **Question Answering**: Ask questions and get answers with source citations
- **SQLite Storage**: Persistent local database for videos and transcripts
- **RAG-Ready Export**: Chunked JSONL/JSON output with metadata for vector databases
- **CLI Interface**: Simple commands with progress indicators

## Documentation

| Document | Description | Video |
|----------|-------------|-------|
| [v0.2 Presentation](docs/v0.2_presentation.md) | Technical deep-dive slides covering RAG concepts, architecture, and benchmarks | |

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

## Commands Overview

| Command | Description |
|---------|-------------|
| `yt-rag init` | Initialize the database |
| `yt-rag add <url>` | Add a channel or video |
| `yt-rag update` | Run full pipeline |
| `yt-rag ask "<query>"` | Ask a question using RAG |
| `yt-rag chat` | Interactive chat |
| `yt-rag status` | Show database statistics |

---

## Pipeline Steps

The update pipeline consists of 6 steps that can be run individually or together via `yt-rag update`.

### Step 1: Add Content (`add`)

Add YouTube channels or individual videos to track.

**What it does:**
- Parses YouTube URLs (channel, video, or playlist)
- Adds the channel/video to the database for tracking
- For channels, discovers all public videos

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `url` | YouTube URL (channel, video, or playlist) |

**Examples:**
```bash
# Add a channel by handle
yt-rag add https://www.youtube.com/@3blue1brown

# Add a channel by ID
yt-rag add https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw

# Add a single video
yt-rag add https://www.youtube.com/watch?v=kCc8FmEb1nY
```

---

### Step 2: Sync Channels (`sync-channel`)

Discover new videos from all tracked channels.

**What it does:**
- Queries YouTube for new videos from each tracked channel
- Adds newly discovered videos to the database
- Skips videos that are already tracked

**Parameters:**
None

**Examples:**
```bash
# Sync all channels
yt-rag sync-channel
```

---

### Step 3: Refresh Metadata (`refresh-meta`)

Fetch or refresh video and channel metadata from YouTube.

**What it does:**
- Fetches video metadata: title, description, duration, view count, publish date, chapters
- Fetches channel metadata: name, description, subscriber count
- By default, skips videos refreshed within the last day

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `--video` | | Refresh only video metadata |
| `--channel` | | Refresh only channel metadata |
| `--limit` | `-l` | Max videos to refresh |
| `--force` | | Refresh even if metadata exists (ignore freshness check) |

**Examples:**
```bash
# Refresh all metadata (respects 1-day freshness)
yt-rag refresh-meta

# Force refresh all video metadata
yt-rag refresh-meta --video --force

# Refresh only channel metadata
yt-rag refresh-meta --channel

# Refresh metadata for 100 videos
yt-rag refresh-meta --video --limit 100
```

---

### Step 4: Fetch Transcripts (`fetch-transcript`)

Download transcripts for videos that don't have them yet.

**What it does:**
- Downloads auto-generated or manual captions from YouTube
- Stores timestamped segments in the database
- Skips private/premium videos and videos without captions
- Uses parallel workers for faster fetching

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `--limit` | | Max videos to fetch |
| `--workers` | `-w` | Number of parallel workers (default: 50) |

**Examples:**
```bash
# Fetch all pending transcripts
yt-rag fetch-transcript

# Fetch transcripts with 100 parallel workers
yt-rag fetch-transcript -w 100

# Fetch only 50 transcripts
yt-rag fetch-transcript --limit 50
```

---

### Step 5: Process Transcripts (`process-transcript`)

Sectionize videos into chapters and generate summaries.

**What it does:**
- **Sectionize**: Splits transcripts into logical sections
  - Uses YouTube chapters when available
  - Falls back to time-based chunking with LLM-generated titles
- **Summarize**: Generates a summary for each video using LLM

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `video_id` | | Process specific video (optional) |
| `--limit` | `-l` | Max videos to process |
| `--sectionize` | | Only run sectionization |
| `--summarize` | | Only run summarization |
| `--model` | `-m` | Override default LLM model |
| `--openai` | | Use OpenAI API instead of local Ollama |
| `--force` | | Re-process even if already done |

**Examples:**
```bash
# Process all pending videos (sectionize + summarize)
yt-rag process-transcript

# Process a specific video
yt-rag process-transcript VIDEO_ID

# Only sectionize (no summaries)
yt-rag process-transcript --sectionize

# Only generate summaries
yt-rag process-transcript --summarize

# Use OpenAI for processing
yt-rag process-transcript --openai

# Use a specific model
yt-rag process-transcript --model gpt-4o

# Force re-process all videos
yt-rag process-transcript --force --limit 100
```

---

### Step 6: Embed (`embed`)

Build FAISS vector indexes for semantic search.

**What it does:**
- Generates embeddings for all sections and summaries
- Stores vectors in FAISS indexes for fast similarity search
- Supports both local (Ollama) and OpenAI embeddings
- Local and OpenAI indexes are stored separately

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `video_id` | | Embed specific video (optional) |
| `--model` | `-m` | Override embedding model |
| `--rebuild` | | Rebuild entire index from scratch |
| `--force` | | Re-embed existing sections |
| `--no-summaries` | | Skip embedding summaries |
| `--openai` | | Use OpenAI embeddings instead of Ollama |

**Examples:**
```bash
# Embed new sections (incremental)
yt-rag embed

# Embed a specific video
yt-rag embed VIDEO_ID

# Rebuild the entire index
yt-rag embed --rebuild

# Use OpenAI embeddings
yt-rag embed --openai

# Use a specific embedding model
yt-rag embed --model text-embedding-3-large --openai
```

---

### Full Pipeline (`update`)

Run all pipeline steps in sequence.

**What it does:**
1. `sync-channel`: Pull new videos from tracked channels
2. `refresh-meta`: Refresh metadata (skips if refreshed within 1 day)
3. `fetch-transcript`: Fetch transcripts for pending videos
4. `process-transcript`: Sectionize and summarize videos
5. `embed`: Build/update vector index
6. `synonyms generate`: Generate synonyms for search boosting

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `--force-transcript` | | Re-fetch all transcripts |
| `--force-meta` | | Force refresh all metadata |
| `--force-embed` | | Rebuild all embeddings |
| `--force-synonym` | | Regenerate all synonyms |
| `--skip-sync` | | Skip channel sync step |
| `--skip-meta` | | Skip metadata refresh step |
| `--skip-embed` | | Skip embedding step |
| `--skip-synonym` | | Skip synonym generation |
| `--test` | | Test mode: 5 videos per channel |
| `--workers` | `-w` | Parallel workers for transcript fetch (default: 50) |
| `--model` | `-m` | Override default LLM model |
| `--openai` | | Use OpenAI API for all steps |

**Examples:**
```bash
# Run full pipeline with local Ollama
yt-rag update

# Run with OpenAI
yt-rag update --openai

# Test mode (5 videos per channel)
yt-rag update --test

# Skip sync and metadata steps
yt-rag update --skip-sync --skip-meta

# Force rebuild everything
yt-rag update --force-transcript --force-meta --force-embed

# Use specific model
yt-rag update --model qwen3:8b
yt-rag update --openai --model gpt-4o
```

---

## RAG Search

### Ask Questions (`ask`)

Query your video content with RAG (Retrieval-Augmented Generation).

**What it does:**
- Analyzes query intent (entity search, comparison, popularity, etc.)
- Retrieves relevant sections via semantic search
- Generates an answer using LLM with retrieved context
- Returns timestamped links to source videos

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `query` | | Question to ask (required) |
| `--top-k` | `-k` | Number of sources to retrieve (default: 5) |
| `--video` | `-v` | Filter to specific video ID |
| `--channel` | `-c` | Filter to specific channel ID |
| `--no-answer` | | Skip answer generation (search only) |
| `--model` | `-m` | Override default LLM model |
| `--openai` | | Use OpenAI API instead of local Ollama |

**Examples:**
```bash
# Ask a question
yt-rag ask "What is tokenization?"

# Get more sources
yt-rag ask "How does attention work?" -k 10

# Filter by channel
yt-rag ask "Explain neural networks" -c CHANNEL_ID

# Filter by video
yt-rag ask "What are the main points?" -v VIDEO_ID

# Search only (no answer generation)
yt-rag ask "transformer architecture" --no-answer

# Use OpenAI
yt-rag ask "What is backpropagation?" --openai

# Use specific model
yt-rag ask "Explain GPT" --model gpt-4o --openai
```

**Example output:**
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

---

### Interactive Chat (`chat`)

Multi-turn conversations with persistent sessions.

**What it does:**
- Maintains conversation history across turns
- Automatically detects follow-up questions
- Supports session management (create, resume, list)
- Retrieves fresh context for each question

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `--video` | `-v` | Filter to specific video ID |
| `--channel` | `-c` | Filter to specific channel ID |
| `--top-k` | `-k` | Number of sections to retrieve (default: 10) |
| `--model` | `-m` | Override default LLM model |
| `--openai` | | Use OpenAI API instead of local Ollama |
| `--new` | | Start a new chat session |
| `--session` | `-s` | Resume session by ID prefix |
| `--list` | | List recent chat sessions |
| `--history` | | Messages to include for context (default: 10) |

**Examples:**
```bash
# Start or resume chat
yt-rag chat

# Start a new session
yt-rag chat --new

# List recent sessions
yt-rag chat --list

# Resume a specific session
yt-rag chat --session abc123

# Filter to a channel
yt-rag chat -c CHANNEL_ID

# Use OpenAI
yt-rag chat --openai
```

**In-chat commands:**
- `/new` - Start a new session
- `/sessions` - List sessions
- `/rename <title>` - Rename current session
- `exit` or `quit` - Exit chat

---

## Search Enhancement

### Keywords (`keywords`)

Extract and analyze keywords from video transcripts.

**What it does:**
- Analyzes transcripts to extract important keywords
- Shows keyword frequency and distribution
- Optionally saves keywords to database for search boosting

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `--limit` | `-n` | Number of videos to analyze (default: 10) |
| `--top-k` | `-k` | Top keywords to show (default: 50) |
| `--channel` | `-c` | Filter by channel ID |
| `--save` | | Save keywords to database |

**Examples:**
```bash
# Analyze keywords from recent videos
yt-rag keywords

# Analyze more videos
yt-rag keywords -n 50

# Show more keywords
yt-rag keywords -k 100

# Filter by channel
yt-rag keywords -c CHANNEL_ID

# Save to database
yt-rag keywords --save
```

---

### Synonyms (`synonyms`)

Manage synonym mappings for improved search recall.

**What it does:**
- Generates synonym suggestions using LLM
- Manages synonym approval workflow
- Boosts search results when synonyms match

**Actions:**
| Action | Description |
|--------|-------------|
| `list` | List current synonyms (default) |
| `generate` | Generate synonym suggestions using LLM |
| `approve` | Approve a pending synonym |
| `reject` | Reject a pending synonym |
| `add` | Add a manual synonym |
| `remove` | Remove synonyms for keyword(s) |

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `--keyword` | `-k` | Keyword to work with |
| `--synonym` | `-s` | Synonym to add/approve/reject |
| `--pending` | | Show pending synonyms only |
| `--limit` | `-n` | Number of keywords to process (default: 10) |
| `--model` | `-m` | Override default LLM model |
| `--openai` | | Use OpenAI API for generation |

**Examples:**
```bash
# List all synonyms
yt-rag synonyms list

# Show pending synonyms
yt-rag synonyms list --pending

# Generate synonym suggestions
yt-rag synonyms generate

# Generate with OpenAI
yt-rag synonyms generate --openai

# Add a manual synonym
yt-rag synonyms add -k "mpg" -s "fuel economy"

# Approve a suggestion
yt-rag synonyms approve -k "hp" -s "horsepower"

# Reject a suggestion
yt-rag synonyms reject -k "car" -s "automobile"

# Remove all synonyms for keywords
yt-rag synonyms remove car truck vehicle
```

---

## Benchmarking

### Generate Test Cases (`test-generate`)

Automatically generate benchmark test cases from video content.

**What it does:**
- Samples videos from your library
- Uses LLM to extract entities, topics, and comparisons
- Generates test queries with expected results

**Workflow:**
1. `prepare`: Sample videos, save raw data
2. `analyze`: LLM extracts entities/topics/facts
3. `build`: Generate test queries

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `--step` | `-s` | Step to run: prepare, analyze, build, or all (default: all) |
| `--videos` | `-n` | Videos to sample per channel (default: 5) |
| `--limit` | `-l` | Max videos to analyze |
| `--model` | `-m` | LLM model for analysis |
| `--openai` | | Use OpenAI API |

**Examples:**
```bash
# Run full test generation
yt-rag test-generate

# Only prepare (sample videos)
yt-rag test-generate --step=prepare

# Analyze with specific model
yt-rag test-generate --step=analyze --model gpt-4o --openai

# Limit analysis to 20 videos
yt-rag test-generate --limit 20
```

---

### Run Benchmark (`test`)

Run the full RAG benchmark suite.

**What it does:**
- Tests query classification accuracy
- Measures retrieval quality (precision, recall, MRR)
- Validates answer quality using LLM
- Compares results across different models

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `--data` | `-d` | Path to test data JSON file |
| `--output` | `-o` | Save results to JSON file |
| `--model` | `-m` | Override LLM model for RAG pipeline |
| `--openai` | | Use OpenAI API |
| `--validate-openai` | | Also validate with OpenAI (compares validators) |
| `--verbose` | `-v` | Show all results, not just failures |

**Examples:**
```bash
# Run benchmark with local Ollama
yt-rag test

# Run with OpenAI
yt-rag test --openai

# Use specific test data
yt-rag test -d tests/data/my_tests.json

# Save results to file
yt-rag test -o results.json

# Verbose output
yt-rag test -v

# Compare local vs OpenAI validators
yt-rag test --validate-openai
```

---

### Generate Report (`test-report`)

Generate HTML report for benchmark results.

**What it does:**
- Creates detailed HTML report showing test results
- Shows validation results from multiple validators
- Supports filtering by pass/fail/disagree

**Parameters:**
| Parameter | Short | Description |
|-----------|-------|-------------|
| `--results` | `-r` | Path to benchmark results JSON |
| `--tests` | `-t` | Path to benchmark tests JSON |
| `--output` | `-o` | Output HTML file path |
| `--filter` | | Filter: pass, fail, disagree, empty, meta |

**Examples:**
```bash
# Generate default report
yt-rag test-report

# Show only failures
yt-rag test-report --filter=fail

# Show only disagreements between validators
yt-rag test-report --filter=disagree

# Custom output path
yt-rag test-report -o my_report.html
```

---

## Browsing & Export

### List Content (`list`)

List tracked channels and videos.

**Examples:**
```bash
# List all channels
yt-rag list channels

# List all videos
yt-rag list videos
```

### Search Videos (`videos`)

Search and filter videos by metadata.

**Examples:**
```bash
# Search videos by title
yt-rag videos --title "neural network"

# Filter by channel
yt-rag videos --channel CHANNEL_ID
```

### Export (`export`)

Export transcript chunks for external RAG pipelines.

**Examples:**
```bash
# Export to JSONL
yt-rag export -o chunks.jsonl

# Export to JSON
yt-rag export -o chunks.json --format json
```

### Single Transcript (`transcript`)

Export a single video's transcript.

**Examples:**
```bash
# Export transcript to file
yt-rag transcript VIDEO_ID -o transcript.txt

# Print to stdout
yt-rag transcript VIDEO_ID
```

---

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

---

## Requirements

- Python 3.14
- Ollama (default) or OpenAI API key (with `--openai`)

## Development

```bash
# Install dev dependencies
make install

# Run linter + formatter (auto-fix)
make lint

# Run full pipeline
make build

# Run tests
make test

# See all available commands
make help
```

Or manually:

```bash
# Install dependencies
uv sync

# Lint and auto-fix
uv run ruff format src/
uv run ruff check --fix src/
```

## License

MIT
