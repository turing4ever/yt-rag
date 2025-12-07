"""Configuration and paths for yt-rag."""

import os
import tomllib
from pathlib import Path

# Default data directory
DATA_DIR = Path.home() / ".yt-rag"
DB_PATH = DATA_DIR / "db.sqlite"
ENV_PATH = DATA_DIR / ".env"
CONFIG_PATH = DATA_DIR / "config.toml"

# Export defaults
DEFAULT_CHUNK_SIZE = 500  # words
DEFAULT_CHUNK_OVERLAP = 50  # words

# Fetch defaults
DEFAULT_FETCH_WORKERS = 50

# yt-dlp rate limiting (max delay in seconds between requests)
# Actual delay is random between 0.1 and this value
# Override with YT_DLP_MAX_DELAY in .env
DEFAULT_YT_DLP_MAX_DELAY = 0.5

# Metadata freshness: skip refresh if updated within this many days
METADATA_FRESHNESS_DAYS = 1

# Sectionizing defaults
# 3 minutes is the average chapter length across all human-labeled YouTube chapters
# The actual duration is now calculated dynamically based on video length in chapters.py
DEFAULT_CHUNK_DURATION = 180  # 3 minutes fallback (dynamic calc preferred)

# Proxy config environment variable name
PROXY_URL_ENV = "WEB_PROXY_URL"

# RAG defaults - separate index directories for local vs OpenAI embeddings
FAISS_DIR = DATA_DIR / "faiss"  # OpenAI embeddings (1536 dims)
FAISS_LOCAL_DIR = DATA_DIR / "faiss_local"  # Ollama embeddings (768 dims)
CHAT_HISTORY_FILE = DATA_DIR / "chat_history"

# OpenAI embedding/chat defaults
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_DIMENSION = 1536

# Ollama defaults
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBEDDING_DIMENSION = 768  # nomic-embed-text dimension

# Search defaults
DEFAULT_TOP_K = 10
DEFAULT_TOP_K_OVERSAMPLE = 50
DEFAULT_SCORE_THRESHOLD = 0.3  # For including results in context
# Threshold for "how many about X" counts (works better for brands than categories)
DEFAULT_COUNT_THRESHOLD = 0.6
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1000

# Hybrid search keyword boost weights (added to semantic score)
KEYWORD_BOOST_VIDEO_TITLE = 0.35  # Boost if query term in video title
KEYWORD_BOOST_SECTION_TITLE = 0.20  # Boost if query term in section title
KEYWORD_BOOST_SECTION_CONTENT = 0.10  # Boost if query term in section content


def get_proxy_url() -> str | None:
    """Get proxy URL from environment.

    Returns:
        Proxy URL string if set, None otherwise.
        Example: http://user:pass@proxy.example.com:8080
    """
    load_env_file()
    return os.environ.get(PROXY_URL_ENV)


def ensure_data_dir() -> Path:
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Load environment variables from .env file."""
    path = path or ENV_PATH
    env_vars = {}
    if not path.exists():
        return env_vars

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            env_vars[key] = value
            os.environ.setdefault(key, value)

    return env_vars


def save_env_var(key: str, value: str, path: Path | None = None) -> None:
    """Save or update an environment variable in .env file."""
    path = path or ENV_PATH
    ensure_data_dir()

    existing_lines = []
    key_found = False

    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    existing_lines.append(f"{key}={value}\n")
                    key_found = True
                else:
                    existing_lines.append(line)

    if not key_found:
        existing_lines.append(f"{key}={value}\n")

    with open(path, "w") as f:
        f.writelines(existing_lines)

    os.environ[key] = value


def load_config(path: Path | None = None) -> dict:
    """Load configuration from TOML file."""
    path = path or CONFIG_PATH
    if not path.exists():
        return {}

    with open(path, "rb") as f:
        return tomllib.load(f)


def get_api_key(key_name: str) -> str | None:
    """Get API key from environment (loads .env first)."""
    load_env_file()
    return os.environ.get(key_name)


def get_yt_dlp_delay() -> float:
    """Get random yt-dlp rate limiting delay in seconds.

    Returns a random value between 0.1 and max_delay.
    Override max with YT_DLP_MAX_DELAY in .env file.
    """
    import random

    load_env_file()
    delay_str = os.environ.get("YT_DLP_MAX_DELAY")
    if delay_str:
        try:
            max_delay = float(delay_str)
        except ValueError:
            max_delay = DEFAULT_YT_DLP_MAX_DELAY
    else:
        max_delay = DEFAULT_YT_DLP_MAX_DELAY

    return random.uniform(0.1, max_delay)
