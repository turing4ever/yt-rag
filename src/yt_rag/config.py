"""Configuration and paths for yt-rag."""

import os
import random
from pathlib import Path

# Default data directory
DATA_DIR = Path.home() / ".yt-rag"
DB_PATH = DATA_DIR / "db.sqlite"
ENV_PATH = DATA_DIR / ".env"

# Export defaults
DEFAULT_CHUNK_SIZE = 500  # words
DEFAULT_CHUNK_OVERLAP = 50  # words

# Fetch defaults
DEFAULT_FETCH_WORKERS = 50

# yt-dlp rate limiting (max delay in seconds between requests)
# Actual delay is random between 0.1 and this value
# Override with YT_DLP_MAX_DELAY in .env
DEFAULT_YT_DLP_MAX_DELAY = 0.5

# yt-dlp batch size for async metadata fetching
# Controls how many concurrent requests per batch
# Override with YT_DLP_BATCH_SIZE in .env
DEFAULT_YT_DLP_BATCH_SIZE = 20

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
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"  # For answer generation
DEFAULT_OLLAMA_QUERY_MODEL = "qwen2.5:7b-instruct"  # For query parsing (better accuracy)
DEFAULT_OLLAMA_EMBED_MODEL = "mxbai-embed-large"
OLLAMA_EMBEDDING_DIMENSION = 1024  # mxbai-embed-large dimension

# FAISS GPU detection flag file
GPU_CHECK_FILE = DATA_DIR / ".gpu_checked"

# Search defaults
DEFAULT_TOP_K = 10
DEFAULT_TOP_K_OVERSAMPLE = 50
DEFAULT_SCORE_THRESHOLD = 0.3  # For including results in context
# Threshold for "how many about X" counts (works better for brands than categories)
DEFAULT_COUNT_THRESHOLD = 0.6
DEFAULT_TEMPERATURE = 0.1  # Low temperature for factual RAG responses
DEFAULT_MAX_TOKENS = 1000

# Hybrid search keyword boost weights (added to semantic score)
KEYWORD_BOOST_VIDEO_TITLE = 0.35  # Boost if query term in video title
KEYWORD_BOOST_SECTION_TITLE = 0.20  # Boost if query term in section title
KEYWORD_BOOST_SECTION_CONTENT = 0.10  # Boost if query term in section content

# Default synonyms (used if database is empty or unavailable)
DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "mpg": ["fuel", "efficiency", "economy", "mileage", "consumption"],
    "fuel": ["mpg", "efficiency", "economy", "mileage", "gas"],
    "hp": ["horsepower", "power", "engine"],
    "horsepower": ["hp", "power", "engine"],
    "torque": ["power", "engine", "lb-ft"],
    "price": ["cost", "msrp", "expensive", "cheap", "value"],
    "cost": ["price", "msrp", "expensive", "value"],
    "interior": ["cabin", "inside", "seats", "dashboard"],
    "exterior": ["outside", "body", "styling", "design"],
    "reliability": ["reliable", "dependable", "issues", "problems"],
}

# Video availability values that restrict transcript access
RESTRICTED_AVAILABILITY = frozenset(
    {
        "private",
        "premium_only",
        "subscriber_only",
        "needs_auth",
    }
)


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


def get_api_key(key_name: str) -> str | None:
    """Get API key from environment (loads .env first)."""
    load_env_file()
    return os.environ.get(key_name)


def get_yt_dlp_delay() -> float:
    """Get random yt-dlp rate limiting delay in seconds.

    Returns a random value between 0.1 and max_delay.
    Override max with YT_DLP_MAX_DELAY in .env file.
    """
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


def get_yt_dlp_batch_size() -> int:
    """Get batch size for async yt-dlp metadata fetching.

    Override with YT_DLP_BATCH_SIZE in .env file.
    """
    load_env_file()
    batch_str = os.environ.get("YT_DLP_BATCH_SIZE")
    if batch_str:
        try:
            return int(batch_str)
        except ValueError:
            pass
    return DEFAULT_YT_DLP_BATCH_SIZE


def has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available on the system.

    Uses nvidia-smi to detect CUDA-capable GPUs.
    Returns True if at least one NVIDIA GPU is detected.
    """
    import shutil
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False

    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # nvidia-smi -L lists GPUs like "GPU 0: NVIDIA GeForce RTX 3080 ..."
        return result.returncode == 0 and "GPU" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def get_gpu_free_memory_mb() -> int | None:
    """Get free GPU memory in MB.

    Returns:
        Free memory in MB, or None if unavailable
    """
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def gpu_check_done() -> bool:
    """Check if GPU detection has already been performed."""
    return GPU_CHECK_FILE.exists()


def mark_gpu_check_done() -> None:
    """Mark GPU detection as complete so we don't prompt again."""
    ensure_data_dir()
    GPU_CHECK_FILE.touch()
