"""RAG Service - async interface for CLI and Web."""

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_COUNT_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_EMBED_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_K_OVERSAMPLE,
)
from .db import Database
from .models import ChatMessage, ChatSession, QueryLog
from .openai_client import (
    achat_completion_stream,
    aembed_text,
    aollama_chat_stream,
    aollama_embed_text,
)
from .search import (
    SearchHit,
    QueryType,
    analyze_query_with_llm,
    classify_query,
    compute_relevance_metrics,
    expand_terms_with_synonyms,
    extract_entity_terms,
    filter_with_llm_analysis,
    find_precise_timestamp,
)
from .vectorstore import get_sections_store, get_summaries_store


@dataclass
class VideoHit:
    """A video-level search result from summary search."""

    video_id: str
    video_title: str
    video_url: str
    channel_id: str | None
    channel_name: str | None
    summary: str
    score: float


@dataclass
class AskResult:
    """Result from ask/search operations."""

    query: str
    hits: list[SearchHit]
    answer: str | None
    tokens_embedding: int
    tokens_chat: int
    latency_ms: int


RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on YouTube video transcripts.

CRITICAL: You must ONLY answer based on the provided context. NEVER mention videos, channels, or content that is not explicitly in the context. If the context doesn't contain relevant information, say so - do not guess or make up answers.

IMPORTANT: Write a short paragraph answer only. Do NOT:
- List numbered items
- Include URLs or timestamps
- Format as "Channel | Video Title"
- Repeat the source entries
- Mention any videos not in the provided context

Only discuss topics explicitly mentioned in the user's query. If the user searches a single keyword (like a car model), just say what videos are available - do not describe the video content or add any details.

The system displays sources separately. Keep answers to 1-2 sentences."""

RAG_USER_PROMPT = """Context from video transcripts (found via semantic search and synonym matching):

{context}

Question: {question}"""

RAG_CONVERSATION_PROMPT = """Context from video transcripts:

{context}

Answer the user's question using the context above."""


def _format_context(
    hits: list[SearchHit],
    db: "Database | None" = None,
    max_chars: int = 8000,
    total_matched_videos: int | None = None,
    video_hits: list["VideoHit"] | None = None,
) -> str:
    """Format search hits as context for the LLM.

    Args:
        hits: Section-level search results
        db: Database to fetch all chapters for each video
        max_chars: Max context size
        total_matched_videos: Total number of unique videos matching the query
        video_hits: Video-level summaries for overview context
    """
    parts = []
    total_chars = 0
    seen_videos: set[str] = set()

    # Add library overview at the start
    if db:
        stats = db.get_stats()
        channels = db.list_channels()
        total_in_library = stats["videos_fetched"]

        # Library stats
        overview = f"[Library: {total_in_library} total videos"
        if total_matched_videos is not None and total_matched_videos > 0:
            overview += f", ~{total_matched_videos} related to this topic"
        overview += "]\n"

        # Add channel details
        for ch in channels:
            overview += f"Channel: {ch.name}"
            if ch.handle:
                overview += f" ({ch.handle})"
            overview += "\n"

        overview += "\n"
        parts.append(overview)
        total_chars += len(overview)

    # Add video-level summaries first (for overview context)
    if video_hits:
        video_section = "[Matching Videos Overview]\n"
        for i, vh in enumerate(video_hits, 1):
            seen_videos.add(vh.video_id)
            summary_text = vh.summary[:500] + "..." if len(vh.summary) > 500 else vh.summary
            video_entry = (
                f"{i}. {vh.video_title}\n   URL: {vh.video_url}\n   Summary: {summary_text}\n\n"
            )
            if total_chars + len(video_section) + len(video_entry) > max_chars // 2:
                break  # Reserve space for section details
            video_section += video_entry

        parts.append(video_section)
        total_chars += len(video_section)

    # Add section-level details
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
        if db and hit.video_id not in seen_videos:
            seen_videos.add(hit.video_id)
            all_sections = db.get_sections(hit.video_id)
            if all_sections:
                chapter_titles = [s.title for s in all_sections]
                chapters_line = f"Chapters: {', '.join(chapter_titles)}\n"

        section_text = (
            f"[Detail {i}] Video: {hit.video_title}\n"
            + (f"{meta_line}\n" if meta_line else "")
            + chapters_line
            + f"Section: {hit.section.title}\n"
            f"Timestamp: {hit.timestamp_url}\n"
            f"Content: {hit.section.content}\n"
        )

        if total_chars + len(section_text) > max_chars:
            break

        parts.append(section_text)
        total_chars += len(section_text)

    return "\n---\n".join(parts)


class RAGService:
    """Async RAG service for querying video transcripts.

    This is the shared interface for CLI and Web UI.
    """

    def __init__(self, db_path: Path | None = None, use_local: bool = True):
        """Initialize the service.

        Args:
            db_path: Path to SQLite database. Uses default if not provided.
            use_local: If True, use local Ollama embeddings; if False, use OpenAI.
        """
        self.db = Database(db_path)
        self.db.init()
        self.use_local = use_local
        self._store = None
        self._summaries_store = None

    def close(self):
        """Close database connection."""
        self.db.close()

    @property
    def store(self):
        """Lazy load sections vector store."""
        if self._store is None:
            self._store = get_sections_store(use_local=self.use_local)
        return self._store

    @property
    def summaries_store(self):
        """Lazy load summaries vector store."""
        if self._summaries_store is None:
            self._summaries_store = get_summaries_store(use_local=self.use_local)
        return self._summaries_store

    async def _embed_query(self, query: str) -> list[float]:
        """Embed a query using the appropriate backend."""
        if self.use_local:
            result = await aollama_embed_text(query, DEFAULT_OLLAMA_EMBED_MODEL)
        else:
            result = await aembed_text(query, DEFAULT_EMBEDDING_MODEL)
        return result.embedding

    async def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        video_id: str | None = None,
        channel_id: str | None = None,
    ) -> list[SearchHit]:
        """Search for relevant sections.

        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum similarity score
            video_id: Filter to specific video
            channel_id: Filter to specific channel

        Returns:
            List of SearchHit results
        """
        if self.store.size == 0:
            return []

        # Embed query using appropriate backend
        query_embedding = await self._embed_query(query)

        # Search vector store
        results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_video_id=video_id,
            filter_channel_id=channel_id,
        )

        # Filter by score and build hits
        hits = []
        for result in results:
            if result.score < score_threshold:
                continue

            section = self.db.get_section(result.id)
            if not section:
                continue

            video = self.db.get_video(section.video_id)
            if not video:
                continue

            channel = self.db.get_channel(video.channel_id) if video.channel_id else None

            timestamp = round(section.start_time) if section.start_time else 0
            timestamp_url = f"https://youtube.com/watch?v={video.id}&t={timestamp}s"

            hits.append(
                SearchHit(
                    section=section,
                    video_id=video.id,
                    video_title=video.title,
                    video_url=video.url,
                    channel_id=video.channel_id,
                    channel_name=channel.name if channel else None,
                    host=video.host,
                    tags=video.tags,
                    score=result.score,
                    timestamp_url=timestamp_url,
                    view_count=video.view_count,
                    like_count=video.like_count,
                    published_at=video.published_at,
                )
            )

        return hits

    async def ask(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        video_id: str | None = None,
        channel_id: str | None = None,
        chat_model: str = DEFAULT_CHAT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        use_ollama: bool = True,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        log_query: bool = True,
    ) -> AskResult:
        """Search and generate an answer (non-streaming).

        Args:
            query: Question to ask
            top_k: Number of sources to retrieve
            score_threshold: Minimum similarity score
            video_id: Filter to specific video
            channel_id: Filter to specific channel
            chat_model: Chat model for answer (OpenAI)
            temperature: LLM temperature
            max_tokens: Max tokens in response
            use_ollama: If True, use local Ollama; if False, use OpenAI
            ollama_model: Ollama model for answer generation
            log_query: Whether to log the query

        Returns:
            AskResult with hits and answer
        """
        from .openai_client import achat_completion, aollama_chat_completion

        start_time = time.time()

        # Search - uses the service's configured embedding backend
        query_embedding = await self._embed_query(query)
        tokens_embedding = 0  # Token counting handled by _embed_query

        if self.store.size == 0:
            return AskResult(
                query=query,
                hits=[],
                answer="No content indexed. Run 'yt-rag embed' first.",
                tokens_embedding=0,
                tokens_chat=0,
                latency_ms=0,
            )

        results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_video_id=video_id,
            filter_channel_id=channel_id,
        )

        # Build hits
        hits = []
        for result in results:
            if result.score < score_threshold:
                continue

            section = self.db.get_section(result.id)
            if not section:
                continue

            video = self.db.get_video(section.video_id)
            if not video:
                continue

            channel = self.db.get_channel(video.channel_id) if video.channel_id else None

            timestamp = round(section.start_time) if section.start_time else 0
            timestamp_url = f"https://youtube.com/watch?v={video.id}&t={timestamp}s"

            hits.append(
                SearchHit(
                    section=section,
                    video_id=video.id,
                    video_title=video.title,
                    video_url=video.url,
                    channel_id=video.channel_id,
                    channel_name=channel.name if channel else None,
                    host=video.host,
                    tags=video.tags,
                    score=result.score,
                    timestamp_url=timestamp_url,
                    view_count=video.view_count,
                    like_count=video.like_count,
                    published_at=video.published_at,
                )
            )

        # Generate answer
        if hits:
            context = _format_context(hits, db=self.db)
            user_content = RAG_USER_PROMPT.format(context=context, question=query)
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            if use_ollama:
                chat_result = await aollama_chat_completion(
                    messages=messages,
                    model=ollama_model,
                    temperature=temperature,
                )
            else:
                chat_result = await achat_completion(
                    messages=messages,
                    model=chat_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            answer = chat_result.content
            tokens_chat = chat_result.tokens_input + chat_result.tokens_output
        else:
            answer = "No relevant content found for your query."
            tokens_chat = 0

        latency_ms = int((time.time() - start_time) * 1000)

        # Log query
        if log_query:
            actual_model = ollama_model if use_ollama else chat_model
            embedding_model = DEFAULT_OLLAMA_EMBED_MODEL if self.use_local else DEFAULT_EMBEDDING_MODEL
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
                model_chat=actual_model,
            )
            self.db.add_query_log(log)

        return AskResult(
            query=query,
            hits=hits,
            answer=answer,
            tokens_embedding=tokens_embedding,
            tokens_chat=tokens_chat,
            latency_ms=latency_ms,
        )

    async def ask_stream(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        video_id: str | None = None,
        channel_id: str | None = None,
        chat_model: str = DEFAULT_CHAT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        use_ollama: bool = True,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        conversation_history: list[dict[str, str]] | None = None,
        cached_hits: list[SearchHit] | None = None,
    ) -> AsyncIterator[str | SearchHit | dict]:
        """Search and stream the answer using two-phase retrieval.

        Phase 1: Search video summaries for overview context
        Phase 2: Search sections for detailed content

        Args:
            query: User's question
            conversation_history: Prior messages [{"role": "user"|"assistant", "content": "..."}]
            cached_hits: Previous search results for follow-up queries

        Yields:
            - VideoHit objects from summary search
            - SearchHit objects from section search
            - dict with {"type": "search_done", ...} when search completes
            - dict with {"type": "followup", "cached_hits": [...]} for follow-up questions
            - str chunks of the answer as they stream
            - dict with {"type": "done", "latency_ms": N} at the end
        """
        import logging

        logger = logging.getLogger(__name__)
        start_time = time.time()

        # Analyze query with LLM to detect intent and extract keywords
        llm_analysis = analyze_query_with_llm(query, use_local=use_ollama)

        # Check for META queries (library stats) - LLM classification only
        if llm_analysis and llm_analysis.query_type == QueryType.META:
            logger.info(f"Meta query detected, returning stats without search")

            # Get comprehensive library stats
            stats = self.db.get_stats()
            channels = self.db.list_channels()
            system_info = self.db.get_all_system_info()

            # Get index info
            sections_store = self.store
            summaries_store = self.summaries_store
            index_type = "GPU" if sections_store.use_gpu else "CPU"

            # Build comprehensive stats text
            stats_lines = [
                "**Library Statistics**",
                f"- Videos: {stats.get('videos_fetched', 0):,} (fetched) / {stats.get('videos_total', 0):,} (total)",
                f"- Channels: {len(channels)}",
                f"- Sections: {stats.get('sections', 0):,}",
                f"- Summaries: {stats.get('summaries', 0):,}",
                f"- Segments: {stats.get('segments', 0):,}",
                "",
                "**Index Info**",
                f"- Sections index: {sections_store.size:,} vectors ({sections_store.dimension}d, {index_type})",
                f"- Summaries index: {summaries_store.size:,} vectors ({summaries_store.dimension}d, {index_type})",
                f"- Embedding: {system_info.get('embedding_backend', 'ollama')} ({system_info.get('embedding_model', 'unknown')})",
            ]

            # Add last embed timestamp if available
            if system_info.get("last_embed_at"):
                stats_lines.append(f"- Last indexed: {system_info.get('last_embed_at', 'unknown')[:19]}")

            stats_lines.extend([
                "",
                "**Channels**",
                ", ".join(ch.name for ch in channels[:20]) + ("..." if len(channels) > 20 else ""),
            ])
            stats_text = "\n".join(stats_lines)

            # Signal "search" complete with no results
            yield {
                "type": "search_done",
                "count": 0,
                "video_summaries": 0,
                "total_matched_videos": 0,
                "meta_query": True,
            }

            # Just yield the answer directly, no LLM needed
            yield stats_text

            latency_ms = int((time.time() - start_time) * 1000)
            yield {"type": "done", "latency_ms": latency_ms, "section_hits": []}
            return

        # Handle follow-up questions using cached results
        if llm_analysis and llm_analysis.query_type == QueryType.FOLLOWUP and cached_hits:
            logger.info(f"Follow-up query detected: {llm_analysis.reasoning}")
            # Signal follow-up mode with cached hits
            yield {"type": "followup", "cached_hits": cached_hits}

            # Build context from cached hits for LLM answer
            section_hits = cached_hits[:top_k]
            for hit in section_hits:
                yield hit

            # Signal search completion
            yield {
                "type": "search_done",
                "count": len(section_hits),
                "video_summaries": 0,
                "total_matched_videos": len({h.video_id for h in section_hits}),
            }

            # Generate answer using cached context
            context = _format_context(section_hits, db=self.db)

            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
            ]

            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-10:])  # Last 10 messages for context

            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

            # Stream answer
            if use_ollama:
                async for chunk in aollama_chat_stream(
                    messages=messages,
                    model=ollama_model,
                    temperature=temperature,
                ):
                    yield chunk
            else:
                async for chunk in achat_completion_stream(
                    messages=messages,
                    model=chat_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield chunk

            latency_ms = int((time.time() - start_time) * 1000)
            yield {"type": "done", "latency_ms": latency_ms, "section_hits": section_hits}
            return

        # Handle popularity queries - search by view count instead of semantics
        if llm_analysis and llm_analysis.query_type == QueryType.POPULARITY:
            logger.info(f"Popularity query detected: {llm_analysis.reasoning}")

            # Extract channel filter from keywords if present (e.g., "joe rogan" -> find channel)
            channel_filter = None
            title_filter = None
            if llm_analysis.keywords:
                keyword_str = " ".join(llm_analysis.keywords)

                # Try to find a matching channel by name or handle
                channels = self.db.list_channels()
                for ch in channels:
                    # Check channel name
                    if keyword_str.lower() in ch.name.lower():
                        channel_filter = ch.id
                        break
                    # Check handle (e.g., @PowerfulJRE)
                    if ch.handle and keyword_str.lower() in ch.handle.lower():
                        channel_filter = ch.id
                        break

                # If no channel match, use as title filter to find videos with keywords
                if not channel_filter:
                    title_filter = keyword_str

            # Calculate published_after date from time_filter_days
            published_after = None
            if llm_analysis.time_filter_days:
                from datetime import datetime, timedelta
                published_after = datetime.now() - timedelta(days=llm_analysis.time_filter_days)
                logger.info(f"Time filter: last {llm_analysis.time_filter_days} days (after {published_after.date()})")

            # Get top videos by view count
            top_videos = self.db.get_top_videos_by_views(
                limit=top_k,
                channel_id=channel_filter or channel_id,
                title_contains=title_filter,
                published_after=published_after,
            )

            if not top_videos:
                yield {"type": "error", "message": "No videos with view count data. Run 'yt-rag refresh-meta --video --force' first."}
                return

            # Build SearchHits from top videos (use first section of each)
            hits: list[SearchHit] = []
            for video in top_videos:
                sections = self.db.get_sections(video.id)
                if not sections:
                    continue
                section = sections[0]  # Use first section
                channel = self.db.get_channel(video.channel_id) if video.channel_id else None

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
                    score=1.0,  # Placeholder score
                    timestamp_url=timestamp_url,
                    view_count=video.view_count,
                    like_count=video.like_count,
                    published_at=video.published_at,
                )
                hits.append(hit)
                yield hit

            # Signal search completion
            yield {
                "type": "search_done",
                "count": len(hits),
                "video_summaries": 0,
                "total_matched_videos": len(hits),
            }

            # Generate answer
            if hits:
                context = _format_context(hits, db=self.db)
                messages = [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                ]
                if conversation_history:
                    messages.extend(conversation_history[-10:])
                messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

                if use_ollama:
                    async for chunk in aollama_chat_stream(
                        messages=messages,
                        model=ollama_model,
                        temperature=temperature,
                    ):
                        yield chunk
                else:
                    async for chunk in achat_completion_stream(
                        messages=messages,
                        model=chat_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        yield chunk

            latency_ms = int((time.time() - start_time) * 1000)
            yield {"type": "done", "latency_ms": latency_ms, "section_hits": hits}
            return

        # Regular search flow
        # Search phase - embed query using appropriate backend
        query_embedding = await self._embed_query(query)

        if self.store.size == 0:
            yield {"type": "error", "message": "No content indexed. Run 'yt-rag embed' first."}
            return

        # Phase 1: Search video summaries for overview (if index exists)
        video_hits: list[VideoHit] = []
        total_summary_matches = 0  # Count of ALL videos whose summaries match
        if self.summaries_store.size > 0:
            # Search ALL summaries to get accurate count
            all_summary_results = self.summaries_store.search(
                query_embedding=query_embedding,
                top_k=self.summaries_store.size,  # Search all
                filter_video_id=video_id,
                filter_channel_id=channel_id,
            )

            # Count all matching summaries above COUNT threshold (higher for accuracy)
            for result in all_summary_results:
                if result.score >= DEFAULT_COUNT_THRESHOLD:
                    total_summary_matches += 1

            # Only yield top 10 for context
            for result in all_summary_results[:10]:
                if result.score < score_threshold:
                    continue

                video = self.db.get_video(result.video_id)
                if not video:
                    continue

                summary = self.db.get_summary(result.video_id)
                if not summary:
                    continue

                channel = self.db.get_channel(video.channel_id) if video.channel_id else None

                video_hit = VideoHit(
                    video_id=video.id,
                    video_title=video.title,
                    video_url=video.url,
                    channel_id=video.channel_id,
                    channel_name=channel.name if channel else None,
                    summary=summary.summary,
                    score=result.score,
                )
                video_hits.append(video_hit)
                yield video_hit

        # Classify query for retrieval strategy
        query_type = classify_query(query)

        # Phase 2: Search sections for detailed content
        # Do a broad search to count ALL matching videos
        broad_results = self.store.search(
            query_embedding=query_embedding,
            top_k=self.store.size,  # Search all vectors
            filter_video_id=video_id,
            filter_channel_id=channel_id,
        )

        # Count unique videos above COUNT threshold (higher for accurate counting)
        all_matched_video_ids: set[str] = set()
        for result in broad_results:
            if result.score < DEFAULT_COUNT_THRESHOLD:
                continue  # Don't break - results may not be sorted by score
            all_matched_video_ids.add(result.video_id)

        total_matched_videos = len(all_matched_video_ids)

        # Phase 2b: Hybrid search - apply relevance scoring and re-rank
        # Take oversampled pool for re-ranking
        oversample_k = max(top_k, DEFAULT_TOP_K_OVERSAMPLE)
        if oversample_k < len(broad_results):
            candidate_results = broad_results[:oversample_k]
        else:
            candidate_results = broad_results

        # Filter by score threshold (use lower threshold for entity searches)
        min_threshold = score_threshold * 0.7 if query_type == QueryType.ENTITY else score_threshold
        candidate_results = [r for r in candidate_results if r.score >= min_threshold]

        # Use LLM analysis (already done above, or do it now if not a follow-up)
        # Filter to only results with keyword/synonym matches to avoid semantic drift
        if llm_analysis:
            # Use the analysis from earlier
            candidate_results, _ = filter_with_llm_analysis(
                query, candidate_results, self.db, use_local=use_ollama
            )
            query_type = llm_analysis.query_type
            logger.info(
                f"LLM analysis: type={query_type.value}, "
                f"keywords={llm_analysis.keywords}, "
                f"reasoning={llm_analysis.reasoning!r}"
            )
        else:
            # LLM analysis failed earlier, try again
            candidate_results, llm_analysis = filter_with_llm_analysis(
                query, candidate_results, self.db, use_local=use_ollama
            )
            if llm_analysis:
                query_type = llm_analysis.query_type

        # Build hits with relevance-based scores
        candidate_hits: list[tuple[float, SearchHit]] = []  # (score, hit)

        for result in candidate_results:
            section = self.db.get_section(result.id)
            if not section:
                continue

            video = self.db.get_video(section.video_id)
            if not video:
                continue

            channel = self.db.get_channel(video.channel_id) if video.channel_id else None

            # Compute relevance metrics (multiplicative + additive scoring)
            metrics = compute_relevance_metrics(
                query=query,
                query_type=query_type,
                semantic_score=result.score,
                video_title=video.title,
                section_title=section.title,
                section_content=section.content,
            )

            # Find precise timestamp within section where query terms appear
            precise_ts = find_precise_timestamp(query, section, self.db)
            timestamp = round(precise_ts) if precise_ts else (round(section.start_time) if section.start_time else 0)
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
                score=metrics.final_score,  # Use new relevance-based score
                timestamp_url=timestamp_url,
                metrics=metrics,
                view_count=video.view_count,
                like_count=video.like_count,
                published_at=video.published_at,
            )
            candidate_hits.append((metrics.final_score, hit))

        # Sort by relevance score (descending) and take top_k
        candidate_hits.sort(key=lambda x: x[0], reverse=True)
        hits = []
        for _, hit in candidate_hits[:top_k]:
            hits.append(hit)
            yield hit

        yield {
            "type": "search_done",
            "count": len(hits),
            "total_videos": total_matched_videos,
            "video_summaries": len(video_hits),
            "total_summary_matches": total_summary_matches,  # Accurate count for "how many about X"
        }

        # Generate streaming answer with both video summaries and section details
        if hits or video_hits:
            # Filter video summaries to only include videos that have relevant sections
            # This prevents irrelevant summary matches from polluting the context
            section_video_ids = {hit.video_id for hit in hits}
            filtered_video_hits = [vh for vh in video_hits if vh.video_id in section_video_ids]

            context = _format_context(
                hits,
                db=self.db,
                total_matched_videos=total_summary_matches,  # Use summary count - more accurate
                video_hits=filtered_video_hits,
            )

            # Enhance "how many" questions with the actual count
            enhanced_query = query
            query_lower = query.lower()
            if "how many" in query_lower:
                total_in_library = self.db.get_stats()["videos_fetched"]
                if total_summary_matches > 0:
                    # Topic-specific count
                    enhanced_query = (
                        f"The semantic search found approximately {total_summary_matches} "
                        f"videos related to this topic (out of {total_in_library} total). "
                        "Please confirm this count and provide some examples from the context."
                    )
                else:
                    # General "how many videos?" - use total
                    enhanced_query = (
                        f"There are {total_in_library} videos in total in the library. "
                        "Please confirm this."
                    )

            # Build messages with optional conversation history
            messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

            if conversation_history:
                # Include prior conversation turns
                # Inject context as a system message before the conversation
                messages.append(
                    {
                        "role": "system",
                        "content": RAG_CONVERSATION_PROMPT.format(context=context),
                    }
                )
                messages.extend(conversation_history)
                # Add the current query
                messages.append({"role": "user", "content": enhanced_query})
            else:
                # Single-turn: use original format
                messages.append(
                    {
                        "role": "user",
                        "content": RAG_USER_PROMPT.format(context=context, question=enhanced_query),
                    }
                )

            # Use Ollama or OpenAI for chat
            if use_ollama:
                async for chunk in aollama_chat_stream(
                    messages=messages,
                    model=ollama_model,
                    temperature=temperature,
                ):
                    yield chunk
            else:
                async for chunk in achat_completion_stream(
                    messages=messages,
                    model=chat_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield chunk
        else:
            yield "No relevant content found for your query."

        latency_ms = int((time.time() - start_time) * 1000)
        yield {"type": "done", "latency_ms": latency_ms, "section_hits": hits}

    def get_status(self) -> dict:
        """Get database and index status."""
        stats = self.db.get_stats()
        stats["index_vectors"] = self.store.size if self._store else 0
        return stats


# Default number of conversation turns to include for context
DEFAULT_HISTORY_LIMIT = 10


class ChatSessionManager:
    """Manages persistent chat sessions with conversation history.

    Designed to work with both CLI and future web UI.
    """

    def __init__(self, db: Database):
        """Initialize with database connection."""
        self.db = db
        self._current_session: ChatSession | None = None

    @property
    def current_session(self) -> ChatSession | None:
        """Get the current active session."""
        return self._current_session

    def create_session(
        self,
        title: str | None = None,
        video_id: str | None = None,
        channel_id: str | None = None,
    ) -> ChatSession:
        """Create a new chat session.

        Args:
            title: Session title. Defaults to "New Chat" (will be updated on first query).
            video_id: Optional filter to specific video.
            channel_id: Optional filter to specific channel.

        Returns:
            The created ChatSession.
        """
        session_id = str(uuid.uuid4())
        session = self.db.create_chat_session(
            session_id=session_id,
            title=title or "New Chat",
            video_id=video_id,
            channel_id=channel_id,
        )
        self._current_session = session
        return session

    def load_session(self, session_id: str) -> ChatSession | None:
        """Load an existing session by ID.

        Args:
            session_id: Session UUID or prefix.

        Returns:
            The session if found, None otherwise.
        """
        # Try exact match first
        session = self.db.get_chat_session(session_id)
        if session:
            self._current_session = session
            return session

        # Try prefix match
        sessions = self.db.list_chat_sessions(limit=100)
        for s in sessions:
            if s.id.startswith(session_id):
                self._current_session = s
                return s

        return None

    def get_most_recent_session(self) -> ChatSession | None:
        """Get the most recently updated session."""
        sessions = self.db.list_chat_sessions(limit=1)
        if sessions:
            self._current_session = sessions[0]
            return sessions[0]
        return None

    def list_sessions(self, limit: int = 20) -> list[ChatSession]:
        """List recent sessions."""
        return self.db.list_chat_sessions(limit=limit)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        session = self.db.get_chat_session(session_id)
        if session:
            self.db.delete_chat_session(session_id)
            if self._current_session and self._current_session.id == session_id:
                self._current_session = None
            return True
        return False

    def add_user_message(self, content: str) -> ChatMessage:
        """Add a user message to the current session.

        If this is the first message in a "New Chat" session,
        auto-generates a title from the query.
        """
        if not self._current_session:
            raise ValueError("No active session. Call create_session() first.")

        message_id = str(uuid.uuid4())
        message = self.db.add_chat_message(
            message_id=message_id,
            session_id=self._current_session.id,
            role="user",
            content=content,
        )

        # Auto-generate title from first user message
        if self._current_session.title == "New Chat":
            title = self._generate_title(content)
            self.db.update_chat_session_title(self._current_session.id, title)
            self._current_session = self.db.get_chat_session(self._current_session.id)

        return message

    def add_assistant_message(self, content: str) -> ChatMessage:
        """Add an assistant message to the current session."""
        if not self._current_session:
            raise ValueError("No active session. Call create_session() first.")

        message_id = str(uuid.uuid4())
        return self.db.add_chat_message(
            message_id=message_id,
            session_id=self._current_session.id,
            role="assistant",
            content=content,
        )

    def get_history(self, limit: int | None = None) -> list[ChatMessage]:
        """Get conversation history for the current session.

        Args:
            limit: Max number of messages (most recent). None for all.

        Returns:
            List of ChatMessage, oldest first.
        """
        if not self._current_session:
            return []
        return self.db.get_chat_messages(self._current_session.id, limit=limit)

    def get_messages_for_llm(self, limit: int = DEFAULT_HISTORY_LIMIT) -> list[dict[str, str]]:
        """Get conversation history formatted for LLM API.

        Args:
            limit: Max number of messages to include.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        messages = self.get_history(limit=limit)
        return [{"role": m.role, "content": m.content} for m in messages]

    def get_message_count(self) -> int:
        """Get count of messages in current session."""
        if not self._current_session:
            return 0
        return self.db.get_chat_message_count(self._current_session.id)

    def rename_session(self, title: str) -> None:
        """Rename the current session."""
        if not self._current_session:
            raise ValueError("No active session.")
        self.db.update_chat_session_title(self._current_session.id, title)
        self._current_session = self.db.get_chat_session(self._current_session.id)

    def _generate_title(self, query: str, max_length: int = 50) -> str:
        """Generate a session title from the first query.

        Uses a simple heuristic: take the query up to max_length chars.
        For questions, preserves the question structure.
        """
        # Clean up query
        title = query.strip()

        # Remove leading "can you" / "please" etc
        lowered = title.lower()
        for prefix in ["can you ", "please ", "could you ", "tell me "]:
            if lowered.startswith(prefix):
                title = title[len(prefix) :]
                break

        # Truncate intelligently
        if len(title) > max_length:
            # Try to cut at word boundary
            cut = title[:max_length].rfind(" ")
            if cut > max_length // 2:
                title = title[:cut] + "..."
            else:
                title = title[:max_length] + "..."

        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]

        return title
