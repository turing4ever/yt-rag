"""RAG search functionality."""

import time
import uuid
from dataclasses import dataclass

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
)
from .db import Database
from .models import QueryLog, Section
from .openai_client import chat_completion, embed_text
from .vectorstore import get_sections_store


@dataclass
class SearchHit:
    """A single search result with full context."""

    section: Section
    video_title: str
    video_url: str
    channel_name: str | None
    score: float
    timestamp_url: str  # YouTube URL with timestamp


@dataclass
class SearchResponse:
    """Complete search response."""

    query: str
    hits: list[SearchHit]
    answer: str | None
    tokens_embedding: int
    tokens_chat: int
    latency_ms: int


RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on YouTube video \
transcripts. Use the provided context to answer the question accurately.

Guidelines:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Cite sources by mentioning the video title and timestamp
4. Be concise but thorough
5. If multiple videos discuss the topic, synthesize the information

Format your response with clear structure when appropriate."""

RAG_USER_PROMPT = """Context from video transcripts:

{context}

Question: {question}

Please answer based on the context above. Cite the video title and timestamp when referencing \
specific information."""


def format_context(hits: list[SearchHit], max_chars: int = 8000) -> str:
    """Format search hits as context for the LLM."""
    parts = []
    total_chars = 0

    for i, hit in enumerate(hits, 1):
        section_text = (
            f"[Source {i}] Video: {hit.video_title}\n"
            f"Section: {hit.section.title}\n"
            f"Timestamp: {hit.timestamp_url}\n"
            f"Content: {hit.section.content}\n"
        )

        if total_chars + len(section_text) > max_chars:
            break

        parts.append(section_text)
        total_chars += len(section_text)

    return "\n---\n".join(parts)


def search(
    query: str,
    db: Database | None = None,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    video_id: str | None = None,
    channel_id: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    generate_answer: bool = True,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    log_query: bool = True,
) -> SearchResponse:
    """Search video transcripts and optionally generate an answer.

    Args:
        query: Search query
        db: Database instance
        top_k: Number of results to return
        score_threshold: Minimum similarity score
        video_id: Filter to specific video
        channel_id: Filter to specific channel
        embedding_model: Model for query embedding
        generate_answer: Whether to generate LLM answer
        chat_model: Model for answer generation
        temperature: LLM temperature
        max_tokens: Max tokens in response
        log_query: Whether to log the query

    Returns:
        SearchResponse with hits and optional answer
    """
    start_time = time.time()

    if db is None:
        db = Database()
        db.init()

    # Get vector store
    store = get_sections_store()
    if store.size == 0:
        return SearchResponse(
            query=query,
            hits=[],
            answer="No content has been indexed yet. Run 'yt-rag embed' first.",
            tokens_embedding=0,
            tokens_chat=0,
            latency_ms=0,
        )

    # Embed query
    embed_result = embed_text(query, embedding_model)
    tokens_embedding = embed_result.tokens_used

    # Search vector store
    results = store.search(
        query_embedding=embed_result.embedding,
        top_k=top_k,
        filter_video_id=video_id,
        filter_channel_id=channel_id,
    )

    # Filter by score threshold
    results = [r for r in results if r.score >= score_threshold]

    # Build search hits with full context
    hits = []
    for result in results:
        section = db.get_section(result.id)
        if not section:
            continue

        video = db.get_video(section.video_id)
        if not video:
            continue

        channel = db.get_channel(video.channel_id) if video.channel_id else None

        # Build timestamp URL
        timestamp = int(section.start_time) if section.start_time else 0
        timestamp_url = f"https://youtube.com/watch?v={video.id}&t={timestamp}s"

        hits.append(
            SearchHit(
                section=section,
                video_title=video.title,
                video_url=video.url,
                channel_name=channel.name if channel else None,
                score=result.score,
                timestamp_url=timestamp_url,
            )
        )

    # Generate answer if requested and we have hits
    answer = None
    tokens_chat = 0

    if generate_answer and hits:
        context = format_context(hits)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": RAG_USER_PROMPT.format(context=context, question=query)},
        ]

        chat_result = chat_completion(
            messages=messages,
            model=chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = chat_result.content
        tokens_chat = chat_result.tokens_input + chat_result.tokens_output
    elif generate_answer and not hits:
        answer = "No relevant content found for your query."

    latency_ms = int((time.time() - start_time) * 1000)

    # Log query
    if log_query:
        log = QueryLog(
            id=str(uuid.uuid4()),
            query=query,
            scope_type="video" if video_id else ("channel" if channel_id else None),
            scope_id=video_id or channel_id,
            retrieved_ids=[h.section.id for h in hits],
            retrieved_scores=[h.score for h in hits],
            answer=answer,
            latency_ms=latency_ms,
            tokens_embedding=tokens_embedding,
            tokens_chat=tokens_chat,
            model_embedding=embedding_model,
            model_chat=chat_model if generate_answer else None,
        )
        db.add_query_log(log)

    return SearchResponse(
        query=query,
        hits=hits,
        answer=answer,
        tokens_embedding=tokens_embedding,
        tokens_chat=tokens_chat,
        latency_ms=latency_ms,
    )


def search_only(
    query: str,
    db: Database | None = None,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    video_id: str | None = None,
    channel_id: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> SearchResponse:
    """Search without generating an answer (cheaper, faster)."""
    return search(
        query=query,
        db=db,
        top_k=top_k,
        score_threshold=score_threshold,
        video_id=video_id,
        channel_id=channel_id,
        embedding_model=embedding_model,
        generate_answer=False,
        log_query=True,
    )
