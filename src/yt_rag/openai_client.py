"""OpenAI API client for embeddings and chat."""

from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    get_api_key,
)


class OpenAIError(Exception):
    """Error from OpenAI API."""

    pass


class OpenAIKeyMissing(OpenAIError):
    """OpenAI API key not configured."""

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


def _should_retry(exc: BaseException) -> bool:
    """Check if exception should be retried."""
    if isinstance(exc, OpenAIKeyMissing):
        return False
    return True


@retry(
    retry=_should_retry,
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
    retry=_should_retry,
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
    retry=_should_retry,
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
