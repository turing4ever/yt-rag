"""FAISS vector store for semantic search."""

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .config import (
    FAISS_DIR,
    FAISS_LOCAL_DIR,
    OLLAMA_EMBEDDING_DIMENSION,
    OPENAI_EMBEDDING_DIMENSION,
    ensure_data_dir,
)


@dataclass
class VectorMetadata:
    """Metadata for a single vector in the index."""

    id: str  # section_id or summary_id
    video_id: str
    channel_id: str | None
    type: str  # 'section' or 'summary'


@dataclass
class SearchResult:
    """Result from a vector search."""

    id: str
    video_id: str
    channel_id: str | None
    score: float
    type: str


class VectorStore:
    """FAISS-based vector store with metadata."""

    def __init__(
        self,
        name: str = "sections",
        dimension: int | None = None,
        index_dir: Path | None = None,
        use_local: bool = True,
    ):
        """Initialize vector store.

        Args:
            name: Name of the index (e.g., 'sections', 'summaries')
            dimension: Vector dimension (auto-detected from use_local if not set)
            index_dir: Directory to store index files (auto-set from use_local if not set)
            use_local: If True, use local Ollama embeddings; if False, use OpenAI
        """
        self.name = name
        self.use_local = use_local

        # Set dimension and directory based on backend
        if dimension is not None:
            self.dimension = dimension
        else:
            self.dimension = OLLAMA_EMBEDDING_DIMENSION if use_local else OPENAI_EMBEDDING_DIMENSION

        if index_dir is not None:
            self.index_dir = index_dir
        else:
            self.index_dir = FAISS_LOCAL_DIR if use_local else FAISS_DIR

        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[VectorMetadata] = []

    @property
    def index_path(self) -> Path:
        return self.index_dir / f"{self.name}.index"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / f"{self.name}_meta.jsonl"

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def _ensure_index(self) -> faiss.IndexFlatIP:
        """Ensure index is created."""
        if self._index is None:
            # Use Inner Product (cosine similarity for normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def add(
        self,
        embeddings: list[list[float]],
        metadata: list[VectorMetadata],
    ) -> None:
        """Add vectors to the index.

        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata, one per vector
        """
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must have same length")

        if not embeddings:
            return

        index = self._ensure_index()

        # Convert to numpy and normalize for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        index.add(vectors)
        self._metadata.extend(metadata)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_video_id: str | None = None,
        filter_channel_id: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_video_id: Only return results from this video
            filter_channel_id: Only return results from this channel

        Returns:
            List of SearchResult sorted by score descending
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Normalize query
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # Over-fetch if filtering, then filter down
        fetch_k = top_k
        if filter_video_id or filter_channel_id:
            fetch_k = min(top_k * 10, self._index.ntotal)

        scores, indices = self._index.search(query, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]

            # Apply filters
            if filter_video_id and meta.video_id != filter_video_id:
                continue
            if filter_channel_id and meta.channel_id != filter_channel_id:
                continue

            results.append(
                SearchResult(
                    id=meta.id,
                    video_id=meta.video_id,
                    channel_id=meta.channel_id,
                    score=float(score),
                    type=meta.type,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def save(self) -> None:
        """Save index and metadata to disk."""
        if self._index is None:
            return

        ensure_data_dir()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))

        # Save metadata as JSONL
        with open(self.metadata_path, "w") as f:
            for meta in self._metadata:
                f.write(
                    json.dumps(
                        {
                            "id": meta.id,
                            "video_id": meta.video_id,
                            "channel_id": meta.channel_id,
                            "type": meta.type,
                        }
                    )
                    + "\n"
                )

    def load(self) -> bool:
        """Load index and metadata from disk.

        Returns:
            True if loaded successfully, False if files don't exist
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        # Load FAISS index
        self._index = faiss.read_index(str(self.index_path))

        # Load metadata
        self._metadata = []
        with open(self.metadata_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self._metadata.append(
                        VectorMetadata(
                            id=data["id"],
                            video_id=data["video_id"],
                            channel_id=data.get("channel_id"),
                            type=data["type"],
                        )
                    )

        # Validate
        if self._index.ntotal != len(self._metadata):
            raise ValueError(
                f"Index size ({self._index.ntotal}) doesn't match "
                f"metadata size ({len(self._metadata)})"
            )

        return True

    def clear(self) -> None:
        """Clear the index and metadata."""
        self._index = None
        self._metadata = []

    def delete_files(self) -> None:
        """Delete index files from disk."""
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

    def get_metadata_by_id(self, id: str) -> VectorMetadata | None:
        """Get metadata for a specific ID."""
        for meta in self._metadata:
            if meta.id == id:
                return meta
        return None

    def get_all_ids(self) -> list[str]:
        """Get all IDs in the index."""
        return [m.id for m in self._metadata]


def get_sections_store(use_local: bool = True) -> VectorStore:
    """Get the sections vector store, loading from disk if exists.

    Args:
        use_local: If True, use local Ollama index; if False, use OpenAI index
    """
    store = VectorStore(name="sections", use_local=use_local)
    store.load()
    return store


def get_summaries_store(use_local: bool = True) -> VectorStore:
    """Get the summaries vector store, loading from disk if exists.

    Args:
        use_local: If True, use local Ollama index; if False, use OpenAI index
    """
    store = VectorStore(name="summaries", use_local=use_local)
    store.load()
    return store
