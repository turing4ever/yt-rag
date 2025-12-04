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

# Proxy config environment variable name
PROXY_URL_ENV = "WEB_PROXY_URL"


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
