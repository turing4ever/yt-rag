"""OpenAI and Ollama API clients for embeddings and chat."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_EMBED_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    OLLAMA_BASE_URL,
    get_api_key,
)


class OpenAIError(Exception):
    """Error from OpenAI API."""

    pass


class OpenAIKeyMissing(OpenAIError):
    """OpenAI API key not configured."""

    pass


class OllamaError(Exception):
    """Error from Ollama API."""

    pass


class OllamaNotRunning(OllamaError):
    """Ollama server is not running."""

    pass


@dataclass
class EmbeddingResult:
    """Result of an embedding request."""

    embedding: list[float]
    tokens_used: int
    model: str


@dataclass
class ChatResult:
    """Result of a chat completion request."""

    content: str
    tokens_input: int
    tokens_output: int
    model: str


def get_client() -> OpenAI:
    """Get OpenAI client, raising if key not configured."""
    api_key = get_api_key("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIKeyMissing(
            "OPENAI_API_KEY not set. Add it to ~/.yt-rag/.env:\nOPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)


def get_async_client() -> AsyncOpenAI:
    """Get async OpenAI client, raising if key not configured."""
    api_key = get_api_key("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIKeyMissing(
            "OPENAI_API_KEY not set. Add it to ~/.yt-rag/.env:\nOPENAI_API_KEY=sk-..."
        )
    return AsyncOpenAI(api_key=api_key)


def _should_retry(exc: BaseException) -> bool:
    """Check if exception should be retried."""
    if isinstance(exc, OpenAIKeyMissing):
        return False
    return True


@retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def embed_text(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingResult:
    """Generate embedding for a single text.

    Args:
        text: Text to embed
        model: Embedding model to use

    Returns:
        EmbeddingResult with embedding vector and token usage
    """
    client = get_client()
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return EmbeddingResult(
        embedding=response.data[0].embedding,
        tokens_used=response.usage.total_tokens,
        model=model,
    )


@retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[EmbeddingResult]:
    """Generate embeddings for multiple texts in a single request.

    Args:
        texts: List of texts to embed
        model: Embedding model to use

    Returns:
        List of EmbeddingResult, one per input text
    """
    if not texts:
        return []

    client = get_client()
    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    # Calculate tokens per text (approximate, API only gives total)
    tokens_per_text = response.usage.total_tokens // len(texts)

    results = []
    for item in response.data:
        results.append(
            EmbeddingResult(
                embedding=item.embedding,
                tokens_used=tokens_per_text,
                model=model,
            )
        )
    return results


@retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def chat_completion(
    messages: list[dict],
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ChatResult:
    """Generate a chat completion.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Chat model to use
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response

    Returns:
        ChatResult with response content and token usage
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return ChatResult(
        content=response.choices[0].message.content or "",
        tokens_input=response.usage.prompt_tokens,
        tokens_output=response.usage.completion_tokens,
        model=model,
    )


def simple_chat(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ChatResult:
    """Simple chat with just a user prompt and optional system message.

    Args:
        prompt: User prompt
        system: Optional system message
        model: Chat model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        ChatResult with response
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat_completion(messages, model, temperature, max_tokens)


def validate_api_key() -> bool:
    """Check if OpenAI API key is valid.

    Returns:
        True if key is valid, False otherwise
    """
    try:
        client = get_client()
        # Make a minimal API call to verify
        client.models.list()
        return True
    except OpenAIKeyMissing:
        return False
    except Exception:
        return False


# =============================================================================
# Async API
# =============================================================================


async def aembed_text(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingResult:
    """Async: Generate embedding for a single text."""
    client = get_async_client()
    response = await client.embeddings.create(
        model=model,
        input=text,
    )
    return EmbeddingResult(
        embedding=response.data[0].embedding,
        tokens_used=response.usage.total_tokens,
        model=model,
    )


async def aembed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[EmbeddingResult]:
    """Async: Generate embeddings for multiple texts in a single request."""
    if not texts:
        return []

    client = get_async_client()
    response = await client.embeddings.create(
        model=model,
        input=texts,
    )

    tokens_per_text = response.usage.total_tokens // len(texts)

    results = []
    for item in response.data:
        results.append(
            EmbeddingResult(
                embedding=item.embedding,
                tokens_used=tokens_per_text,
                model=model,
            )
        )
    return results


async def achat_completion(
    messages: list[dict],
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ChatResult:
    """Async: Generate a chat completion."""
    client = get_async_client()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return ChatResult(
        content=response.choices[0].message.content or "",
        tokens_input=response.usage.prompt_tokens,
        tokens_output=response.usage.completion_tokens,
        model=model,
    )


async def achat_completion_stream(
    messages: list[dict],
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AsyncIterator[str]:
    """Async: Stream chat completion chunks.

    Yields:
        Text chunks as they arrive from the API
    """
    client = get_async_client()
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def asimple_chat(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ChatResult:
    """Async: Simple chat with just a user prompt and optional system message."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return await achat_completion(messages, model, temperature, max_tokens)


async def asimple_chat_stream(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AsyncIterator[str]:
    """Async: Stream simple chat response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    async for chunk in achat_completion_stream(messages, model, temperature, max_tokens):
        yield chunk


# =============================================================================
# Ollama API (Local LLM)
# =============================================================================


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


def get_ollama_models() -> list[str]:
    """Get list of available Ollama models."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except httpx.RequestError as e:
        raise OllamaNotRunning(
            "Ollama server not running. Start it with: sudo systemctl start ollama"
        ) from e


def ollama_chat_completion(
    messages: list[dict],
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatResult:
    """Generate a chat completion using Ollama.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Ollama model to use
        temperature: Sampling temperature

    Returns:
        ChatResult with response content
    """
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return ChatResult(
            content=data["message"]["content"],
            tokens_input=data.get("prompt_eval_count", 0),
            tokens_output=data.get("eval_count", 0),
            model=model,
        )
    except httpx.ConnectError as e:
        raise OllamaNotRunning(
            "Ollama server not running. Start it with: sudo systemctl start ollama"
        ) from e
    except httpx.RequestError as e:
        raise OllamaError(f"Ollama request failed: {e}") from e


async def aollama_chat_completion(
    messages: list[dict],
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatResult:
    """Async: Generate a chat completion using Ollama."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            return ChatResult(
                content=data["message"]["content"],
                tokens_input=data.get("prompt_eval_count", 0),
                tokens_output=data.get("eval_count", 0),
                model=model,
            )
    except httpx.ConnectError as e:
        raise OllamaNotRunning(
            "Ollama server not running. Start it with: sudo systemctl start ollama"
        ) from e
    except httpx.RequestError as e:
        raise OllamaError(f"Ollama request failed: {e}") from e


async def aollama_chat_stream(
    messages: list[dict],
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> AsyncIterator[str]:
    """Async: Stream chat completion from Ollama.

    Yields:
        Text chunks as they arrive from Ollama
    """
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": temperature},
                },
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json

                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
    except httpx.ConnectError as e:
        raise OllamaNotRunning(
            "Ollama server not running. Start it with: sudo systemctl start ollama"
        ) from e
    except httpx.RequestError as e:
        raise OllamaError(f"Ollama request failed: {e}") from e


# =============================================================================
# Ollama Embeddings (Local)
# =============================================================================


def ollama_embed_text(
    text: str,
    model: str = DEFAULT_OLLAMA_EMBED_MODEL,
) -> EmbeddingResult:
    """Generate embedding using Ollama.

    Args:
        text: Text to embed
        model: Ollama embedding model (default: mxbai-embed-large)

    Returns:
        EmbeddingResult with embedding vector
    """
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": text},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return EmbeddingResult(
            embedding=data["embeddings"][0],
            tokens_used=0,  # Ollama doesn't report tokens
            model=model,
        )
    except httpx.ConnectError as e:
        raise OllamaNotRunning(
            "Ollama server not running. Start it with: sudo systemctl start ollama"
        ) from e
    except httpx.RequestError as e:
        raise OllamaError(f"Ollama embedding failed: {e}") from e


def ollama_embed_texts(
    texts: list[str],
    model: str = DEFAULT_OLLAMA_EMBED_MODEL,
) -> list[EmbeddingResult]:
    """Generate embeddings for multiple texts using Ollama.

    Args:
        texts: List of texts to embed
        model: Ollama embedding model (default: mxbai-embed-large)

    Returns:
        List of EmbeddingResult, one per input text
    """
    if not texts:
        return []

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": texts},
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for embedding in data["embeddings"]:
            results.append(
                EmbeddingResult(
                    embedding=embedding,
                    tokens_used=0,
                    model=model,
                )
            )
        return results
    except httpx.ConnectError as e:
        raise OllamaNotRunning(
            "Ollama server not running. Start it with: sudo systemctl start ollama"
        ) from e
    except httpx.RequestError as e:
        raise OllamaError(f"Ollama embedding failed: {e}") from e


async def aollama_embed_text(
    text: str,
    model: str = DEFAULT_OLLAMA_EMBED_MODEL,
) -> EmbeddingResult:
    """Async: Generate embedding using Ollama."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": model, "input": text},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return EmbeddingResult(
                embedding=data["embeddings"][0],
                tokens_used=0,
                model=model,
            )
    except httpx.ConnectError as e:
        raise OllamaNotRunning(
            "Ollama server not running. Start it with: sudo systemctl start ollama"
        ) from e
    except httpx.RequestError as e:
        raise OllamaError(f"Ollama embedding failed: {e}") from e


async def aollama_embed_texts(
    texts: list[str],
    model: str = DEFAULT_OLLAMA_EMBED_MODEL,
) -> list[EmbeddingResult]:
    """Async: Generate embeddings for multiple texts using Ollama."""
    if not texts:
        return []

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": model, "input": texts},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for embedding in data["embeddings"]:
                results.append(
                    EmbeddingResult(
                        embedding=embedding,
                        tokens_used=0,
                        model=model,
                    )
                )
            return results
    except httpx.ConnectError as e:
        raise OllamaNotRunning(
            "Ollama server not running. Start it with: sudo systemctl start ollama"
        ) from e
    except httpx.RequestError as e:
        raise OllamaError(f"Ollama embedding failed: {e}") from e


def get_ollama_embedding_dimension(model: str = DEFAULT_OLLAMA_EMBED_MODEL) -> int:
    """Get the embedding dimension for an Ollama model by making a test call."""
    result = ollama_embed_text("test", model)
    return len(result.embedding)
