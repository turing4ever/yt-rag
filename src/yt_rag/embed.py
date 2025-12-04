"""Embedding and indexing for RAG search."""

from dataclasses import dataclass

from .config import DEFAULT_EMBEDDING_MODEL
from .db import Database
from .models import Section
from .openai_client import embed_texts
from .vectorstore import VectorMetadata, VectorStore, get_sections_store


@dataclass
class EmbedResult:
    """Result of embedding operation."""

    sections_embedded: int
    tokens_used: int


def embed_sections(
    sections: list[Section],
    store: VectorStore,
    db: Database,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 100,
) -> EmbedResult:
    """Embed sections and add to vector store.

    Args:
        sections: List of sections to embed
        store: Vector store to add embeddings to
        db: Database for looking up video/channel info
        model: Embedding model to use
        batch_size: Number of sections to embed per API call

    Returns:
        EmbedResult with counts and token usage
    """
    if not sections:
        return EmbedResult(sections_embedded=0, tokens_used=0)

    total_tokens = 0
    embedded = 0

    # Process in batches
    for i in range(0, len(sections), batch_size):
        batch = sections[i : i + batch_size]

        # Get text content for embedding
        texts = [f"{s.title}\n\n{s.content}" for s in batch]

        # Get embeddings
        results = embed_texts(texts, model)
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

    return EmbedResult(sections_embedded=embedded, tokens_used=total_tokens)


def embed_all_sections(
    db: Database | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 100,
    rebuild: bool = False,
) -> EmbedResult:
    """Embed all sections in the database.

    Args:
        db: Database instance
        model: Embedding model to use
        batch_size: Sections per API call
        rebuild: If True, rebuild index from scratch

    Returns:
        EmbedResult with counts
    """
    if db is None:
        db = Database()
        db.init()

    store = get_sections_store()

    if rebuild:
        store.clear()
        store.delete_files()

    # Get sections that need embedding
    all_sections = db.get_all_sections()

    if not rebuild:
        # Skip already embedded sections
        existing_ids = set(store.get_all_ids())
        all_sections = [s for s in all_sections if s.id not in existing_ids]

    if not all_sections:
        return EmbedResult(sections_embedded=0, tokens_used=0)

    result = embed_sections(all_sections, store, db, model, batch_size)

    # Save index
    store.save()

    return result


def embed_video(
    video_id: str,
    db: Database | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    force: bool = False,
) -> EmbedResult:
    """Embed sections for a single video.

    Args:
        video_id: Video to embed
        db: Database instance
        model: Embedding model
        force: Re-embed even if already done

    Returns:
        EmbedResult
    """
    if db is None:
        db = Database()
        db.init()

    sections = db.get_sections(video_id)
    if not sections:
        return EmbedResult(sections_embedded=0, tokens_used=0)

    store = get_sections_store()

    if not force:
        # Check if already embedded
        existing_ids = set(store.get_all_ids())
        sections = [s for s in sections if s.id not in existing_ids]

    if not sections:
        return EmbedResult(sections_embedded=0, tokens_used=0)

    result = embed_sections(sections, store, db, model)
    store.save()

    return result


def get_index_stats() -> dict:
    """Get statistics about the FAISS index."""
    store = get_sections_store()

    stats = {
        "total_vectors": store.size,
        "index_exists": store.index_path.exists(),
        "metadata_exists": store.metadata_path.exists(),
    }

    if store.size > 0:
        # Count by type
        type_counts = {}
        video_counts = {}
        for meta in store._metadata:
            type_counts[meta.type] = type_counts.get(meta.type, 0) + 1
            video_counts[meta.video_id] = video_counts.get(meta.video_id, 0) + 1

        stats["by_type"] = type_counts
        stats["videos_indexed"] = len(video_counts)

    return stats
