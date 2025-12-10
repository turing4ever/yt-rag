"""Embedding and indexing for RAG search."""

from dataclasses import dataclass

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_EMBED_MODEL
from .db import Database
from .models import Section
from .openai_client import embed_texts, ollama_embed_texts
from .vectorstore import VectorMetadata, VectorStore, get_sections_store, get_summaries_store


@dataclass
class EmbedResult:
    """Result of embedding operation."""

    items_embedded: int  # Sections or summaries embedded
    tokens_used: int


def embed_sections(
    sections: list[Section],
    store: VectorStore,
    db: Database,
    model: str | None = None,
    batch_size: int = 100,
    use_local: bool = True,
) -> EmbedResult:
    """Embed sections and add to vector store.

    Args:
        sections: List of sections to embed
        store: Vector store to add embeddings to
        db: Database for looking up video/channel info
        model: Embedding model to use (auto-selected based on use_local if None)
        batch_size: Number of sections to embed per API call
        use_local: If True, use Ollama local embeddings; if False, use OpenAI

    Returns:
        EmbedResult with counts and token usage
    """
    if not sections:
        return EmbedResult(items_embedded=0, tokens_used=0)

    # Select model based on backend
    if model is None:
        model = DEFAULT_OLLAMA_EMBED_MODEL if use_local else DEFAULT_EMBEDDING_MODEL

    # Select embedding function
    embed_fn = ollama_embed_texts if use_local else embed_texts

    # Use smaller batches for local to avoid memory issues
    if use_local:
        batch_size = min(batch_size, 32)

    total_tokens = 0
    embedded = 0

    # Process in batches
    for i in range(0, len(sections), batch_size):
        batch = sections[i : i + batch_size]

        # Get text content for embedding
        texts = [f"{s.title}\n\n{s.content}" for s in batch]

        # Get embeddings
        results = embed_fn(texts, model)
        total_tokens += sum(r.tokens_used for r in results)

        # Build metadata and add to store
        embeddings = [r.embedding for r in results]
        metadata = []

        for section in batch:
            video = db.get_video(section.video_id)
            channel_id = video.channel_id if video else None

            metadata.append(
                VectorMetadata(
                    id=section.id,
                    video_id=section.video_id,
                    channel_id=channel_id,
                    type="section",
                )
            )

        store.add(embeddings, metadata)
        embedded += len(batch)

    return EmbedResult(items_embedded=embedded, tokens_used=total_tokens)


def embed_all_sections(
    db: Database | None = None,
    model: str | None = None,
    batch_size: int = 100,
    rebuild: bool = False,
    use_local: bool = True,
) -> EmbedResult:
    """Embed all sections in the database.

    Args:
        db: Database instance
        model: Embedding model to use (auto-selected if None)
        batch_size: Sections per API call
        rebuild: If True, rebuild index from scratch
        use_local: If True, use Ollama; if False, use OpenAI

    Returns:
        EmbedResult with counts
    """
    if db is None:
        db = Database()
        db.init()

    store = get_sections_store(use_local=use_local)

    if rebuild:
        store.clear()
        store.delete_files()

    # Get sections that need embedding
    all_sections = db.get_all_sections()

    if not rebuild:
        # Skip already embedded sections
        existing_ids = set(store.get_all_ids())
        all_sections = [s for s in all_sections if s.id not in existing_ids]

    # Always update system info (even if no new sections to embed)
    from datetime import datetime

    actual_model = model or (DEFAULT_OLLAMA_EMBED_MODEL if use_local else DEFAULT_EMBEDDING_MODEL)
    db.set_system_info("embedding_backend", "ollama" if use_local else "openai")
    db.set_system_info("embedding_model", actual_model)
    db.set_system_info("embedding_dimension", str(store.dimension))
    db.set_system_info("sections_index_size", str(store.size))

    if not all_sections:
        return EmbedResult(items_embedded=0, tokens_used=0)

    result = embed_sections(all_sections, store, db, model, batch_size, use_local=use_local)

    # Save index
    store.save()

    # Update last embed time
    db.set_system_info("last_embed_at", datetime.now().isoformat())

    return result


def embed_video(
    video_id: str,
    db: Database | None = None,
    model: str | None = None,
    force: bool = False,
    use_local: bool = True,
) -> EmbedResult:
    """Embed sections for a single video.

    Args:
        video_id: Video to embed
        db: Database instance
        model: Embedding model (auto-selected if None)
        force: Re-embed even if already done
        use_local: If True, use Ollama; if False, use OpenAI

    Returns:
        EmbedResult
    """
    if db is None:
        db = Database()
        db.init()

    sections = db.get_sections(video_id)
    if not sections:
        return EmbedResult(items_embedded=0, tokens_used=0)

    store = get_sections_store(use_local=use_local)

    if not force:
        # Check if already embedded
        existing_ids = set(store.get_all_ids())
        sections = [s for s in sections if s.id not in existing_ids]

    if not sections:
        return EmbedResult(items_embedded=0, tokens_used=0)

    result = embed_sections(sections, store, db, model, use_local=use_local)
    store.save()

    return result


def embed_all_summaries(
    db: Database | None = None,
    model: str | None = None,
    batch_size: int = 100,
    rebuild: bool = False,
    use_local: bool = True,
) -> EmbedResult:
    """Embed all video summaries into a separate index.

    This creates a video-level index for overview queries.

    Args:
        db: Database instance
        model: Embedding model to use (auto-selected if None)
        batch_size: Summaries per API call
        rebuild: If True, rebuild index from scratch
        use_local: If True, use Ollama; if False, use OpenAI

    Returns:
        EmbedResult with counts
    """
    if db is None:
        db = Database()
        db.init()

    # Select model based on backend
    if model is None:
        model = DEFAULT_OLLAMA_EMBED_MODEL if use_local else DEFAULT_EMBEDDING_MODEL

    # Select embedding function
    embed_fn = ollama_embed_texts if use_local else embed_texts

    # Use smaller batches for local
    if use_local:
        batch_size = min(batch_size, 32)

    store = get_summaries_store(use_local=use_local)

    if rebuild:
        store.clear()
        store.delete_files()

    # Get all summaries
    all_summaries = db.get_all_summaries()

    if not rebuild:
        # Skip already embedded summaries
        existing_ids = set(store.get_all_ids())
        all_summaries = [s for s in all_summaries if s.video_id not in existing_ids]

    if not all_summaries:
        return EmbedResult(items_embedded=0, tokens_used=0)

    total_tokens = 0
    embedded = 0

    # Process in batches
    for i in range(0, len(all_summaries), batch_size):
        batch = all_summaries[i : i + batch_size]

        # Get text content for embedding (include video title for context)
        texts = []
        for summary in batch:
            video = db.get_video(summary.video_id)
            title = video.title if video else ""
            texts.append(f"{title}\n\n{summary.summary}")

        # Get embeddings
        results = embed_fn(texts, model)
        total_tokens += sum(r.tokens_used for r in results)

        # Build metadata and add to store
        embeddings = [r.embedding for r in results]
        metadata = []

        for summary in batch:
            video = db.get_video(summary.video_id)
            channel_id = video.channel_id if video else None

            metadata.append(
                VectorMetadata(
                    id=summary.video_id,  # Use video_id as ID for summaries
                    video_id=summary.video_id,
                    channel_id=channel_id,
                    type="summary",
                )
            )

        store.add(embeddings, metadata)
        embedded += len(batch)

    # Save index
    store.save()

    return EmbedResult(items_embedded=embedded, tokens_used=total_tokens)


def get_index_stats(use_local: bool = True) -> dict:
    """Get statistics about both FAISS indexes.

    Args:
        use_local: If True, report on local indexes; if False, report on OpenAI indexes
    """
    sections_store = get_sections_store(use_local=use_local)
    summaries_store = get_summaries_store(use_local=use_local)

    backend = "local" if use_local else "openai"
    stats = {
        "backend": backend,
        "sections_vectors": sections_store.size,
        "summaries_vectors": summaries_store.size,
        "total_vectors": sections_store.size + summaries_store.size,
        "sections_index_exists": sections_store.index_path.exists(),
        "summaries_index_exists": summaries_store.index_path.exists(),
        "index_dir": str(sections_store.index_dir),
    }

    if sections_store.size > 0:
        # Count videos in sections index
        video_counts = {}
        for meta in sections_store._metadata:
            video_counts[meta.video_id] = video_counts.get(meta.video_id, 0) + 1
        stats["videos_with_sections"] = len(video_counts)

    if summaries_store.size > 0:
        stats["videos_with_summaries"] = summaries_store.size

    return stats
