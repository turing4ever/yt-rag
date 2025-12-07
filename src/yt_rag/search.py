"""RAG search functionality."""

import re
import time
import uuid
from dataclasses import dataclass

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_EMBED_MODEL,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_OVERSAMPLE,
    KEYWORD_BOOST_SECTION_CONTENT,
    KEYWORD_BOOST_SECTION_TITLE,
    KEYWORD_BOOST_VIDEO_TITLE,
)
from .db import Database
from .models import QueryLog, Section
from .openai_client import chat_completion, embed_text, ollama_embed_text
from .vectorstore import get_sections_store


@dataclass
class SearchHit:
    """A single search result with full context."""

    section: Section
    video_id: str  # Video ID for deduplication
    video_title: str
    video_url: str
    channel_id: str | None  # Channel ID for filtering
    channel_name: str | None
    host: str | None  # Video host/creator
    tags: list[str] | None  # Video tags
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


RAG_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on YouTube video transcripts.
Use the provided context to answer the question accurately.

Guidelines:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. ALWAYS cite sources with clickable links using markdown: [Video Title](URL)
4. Be concise but thorough
5. If multiple videos discuss the topic, synthesize and cite each source
6. The [Library Overview] section contains database stats - use it for questions about \
how many channels/videos exist, but don't cite it as a source

Example: The CX-70 has three drivetrain options \
([2025 Mazda CX-70 Review](https://youtube.com/watch?v=abc&t=117s))."""

RAG_USER_PROMPT = """\
Context from video transcripts:

{context}

Question: {question}

Answer using the context above. Cite each fact with a markdown link: \
[Video Title](timestamp_url)."""


def _compute_keyword_boost(
    query: str,
    video_title: str,
    section_title: str,
    section_content: str,
) -> float:
    """Compute keyword boost for hybrid search re-ranking.

    Extracts terms from the query and checks for their presence in:
    - Video title (highest boost)
    - Section title (medium boost)
    - Section content (lower boost)

    Args:
        query: The search query
        video_title: Video title to check
        section_title: Section title to check
        section_content: Section content to check

    Returns:
        Total boost value to add to semantic score
    """
    # Extract meaningful terms (2+ chars, alphanumeric)
    terms = re.findall(r"\b[a-zA-Z0-9]{2,}\b", query.lower())
    if not terms:
        return 0.0

    # Normalize texts for case-insensitive matching
    video_title_lower = video_title.lower()
    section_title_lower = section_title.lower()
    section_content_lower = section_content.lower()

    boost = 0.0

    for term in terms:
        # Check each level - only count once per term per level
        if term in video_title_lower:
            boost += KEYWORD_BOOST_VIDEO_TITLE
        if term in section_title_lower:
            boost += KEYWORD_BOOST_SECTION_TITLE
        if term in section_content_lower:
            boost += KEYWORD_BOOST_SECTION_CONTENT

    return boost


def format_context(
    hits: list[SearchHit],
    db: Database | None = None,
    max_chars: int = 8000,
) -> str:
    """Format search hits as context for the LLM.

    Args:
        hits: Search results
        db: Database to fetch all chapters for each video
        max_chars: Max context size
    """
    parts = []
    total_chars = 0
    seen_videos: set[str] = set()

    # Add library overview at the start
    if db:
        stats = db.get_stats()
        channels = db.list_channels()
        channel_names = [c.name for c in channels]
        channels_str = ", ".join(channel_names) if channel_names else "None"
        overview = (
            f"[Library Overview]\n"
            f"Channels: {channels_str} ({stats['channels']} total)\n"
            f"Videos: {stats['videos_fetched']} indexed\n"
            f"Sections: {stats['sections']}\n\n"
        )
        parts.append(overview)
        total_chars += len(overview)

    for i, hit in enumerate(hits, 1):
        # Build metadata line
        meta_parts = []
        if hit.channel_name:
            meta_parts.append(f"Channel: {hit.channel_name}")
        if hit.host and hit.host != hit.channel_name:
            meta_parts.append(f"Host: {hit.host}")
        if hit.tags:
            meta_parts.append(f"Tags: {', '.join(hit.tags[:5])}")
        meta_line = " | ".join(meta_parts) if meta_parts else ""

        # Get all chapters for this video (first time we see it)
        chapters_line = ""
        video_id = hit.section.video_id
        if db and video_id not in seen_videos:
            seen_videos.add(video_id)
            all_sections = db.get_sections(video_id)
            if all_sections:
                chapter_titles = [s.title for s in all_sections]
                chapters_line = f"Chapters: {', '.join(chapter_titles)}\n"

        section_text = (
            f"[Source {i}] Video: {hit.video_title}\n"
            + (f"{meta_line}\n" if meta_line else "")
            + chapters_line
            + f"Current Section: {hit.section.title}\n"
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
    embedding_model: str | None = None,
    generate_answer: bool = True,
    chat_model: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    log_query: bool = True,
    use_local: bool = True,
) -> SearchResponse:
    """Search video transcripts and optionally generate an answer.

    Args:
        query: Search query
        db: Database instance
        top_k: Number of results to return
        score_threshold: Minimum similarity score
        video_id: Filter to specific video
        channel_id: Filter to specific channel
        embedding_model: Model for query embedding (auto-selected if None)
        generate_answer: Whether to generate LLM answer
        chat_model: Model for answer generation
        temperature: LLM temperature
        max_tokens: Max tokens in response
        log_query: Whether to log the query
        use_local: If True, use Ollama for embeddings; if False, use OpenAI

    Returns:
        SearchResponse with hits and optional answer
    """
    start_time = time.time()

    if db is None:
        db = Database()
        db.init()

    # Select embedding model based on backend
    if embedding_model is None:
        embedding_model = DEFAULT_OLLAMA_EMBED_MODEL if use_local else DEFAULT_EMBEDDING_MODEL

    # Get vector store for the appropriate backend
    store = get_sections_store(use_local=use_local)
    if store.size == 0:
        return SearchResponse(
            query=query,
            hits=[],
            answer="No content has been indexed yet. Run 'yt-rag embed' first.",
            tokens_embedding=0,
            tokens_chat=0,
            latency_ms=0,
        )

    # Embed query using the appropriate backend
    if use_local:
        embed_result = ollama_embed_text(query, embedding_model)
    else:
        embed_result = embed_text(query, embedding_model)
    tokens_embedding = embed_result.tokens_used

    # Search vector store with oversampling for keyword re-ranking
    oversample_k = max(top_k, DEFAULT_TOP_K_OVERSAMPLE)
    results = store.search(
        query_embedding=embed_result.embedding,
        top_k=oversample_k,
        filter_video_id=video_id,
        filter_channel_id=channel_id,
    )

    # Filter by score threshold
    results = [r for r in results if r.score >= score_threshold]

    # Build search hits with keyword-boosted scores
    candidate_hits: list[tuple[float, SearchHit]] = []
    for result in results:
        section = db.get_section(result.id)
        if not section:
            continue

        video = db.get_video(section.video_id)
        if not video:
            continue

        channel = db.get_channel(video.channel_id) if video.channel_id else None

        # Compute keyword boost for hybrid search
        keyword_boost = _compute_keyword_boost(
            query=query,
            video_title=video.title,
            section_title=section.title,
            section_content=section.content,
        )
        boosted_score = result.score + keyword_boost

        # Build timestamp URL (use round for better precision)
        timestamp = round(section.start_time) if section.start_time else 0
        timestamp_url = f"https://youtube.com/watch?v={video.id}&t={timestamp}s"

        hit = SearchHit(
            section=section,
            video_id=video.id,
            video_title=video.title,
            video_url=video.url,
            channel_id=video.channel_id,
            channel_name=channel.name if channel else None,
            host=video.host,
            tags=video.tags,
            score=boosted_score,  # Use boosted score
            timestamp_url=timestamp_url,
        )
        candidate_hits.append((boosted_score, hit))

    # Sort by boosted score (descending) and take top_k
    candidate_hits.sort(key=lambda x: x[0], reverse=True)
    hits = [hit for _, hit in candidate_hits[:top_k]]

    # Generate answer if requested and we have hits
    answer = None
    tokens_chat = 0

    if generate_answer and hits:
        context = format_context(hits, db=db)
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
    embedding_model: str | None = None,
    use_local: bool = True,
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
        use_local=use_local,
    )
