"""FAISS vector store for semantic search."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Conditional FAISS import - prefer GPU if available
try:
    import faiss

    # Check if GPU is available (faiss-gpu installed and CUDA working)
    _num_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    FAISS_GPU_AVAILABLE = _num_gpus > 0
except ImportError:
    # This shouldn't happen since faiss-cpu is a dependency, but handle gracefully
    raise ImportError(
        "FAISS is not installed. Install with: pip install faiss-cpu "
        "or pip install faiss-gpu (requires CUDA)"
    )


def _check_gpu_memory_available(min_free_mb: int = 512) -> bool:
    """Check if GPU has enough free memory for FAISS operations.

    Args:
        min_free_mb: Minimum free memory in MB required

    Returns:
        True if enough memory is available, False otherwise
    """
    if not FAISS_GPU_AVAILABLE:
        return False
    free_mb = get_gpu_free_memory_mb()
    if free_mb is not None:
        return free_mb >= min_free_mb
    return True  # Assume available if we can't check

logger = logging.getLogger(__name__)


def check_gpu_upgrade_prompt() -> None:
    """Check if GPU is available but faiss-gpu is not installed.

    On first run, if NVIDIA GPU is detected but we're using faiss-cpu,
    suggest upgrading to faiss-gpu for better performance.
    Only prompts once (stores flag to avoid repeated prompts).
    """
    from .config import gpu_check_done, has_nvidia_gpu, mark_gpu_check_done

    if gpu_check_done():
        return

    mark_gpu_check_done()

    # Only suggest if GPU hardware exists but faiss-gpu not installed
    if not FAISS_GPU_AVAILABLE and has_nvidia_gpu():
        logger.info(
            "NVIDIA GPU detected but faiss-gpu is not installed. "
            "For faster vector search, consider upgrading:\n"
            "  pip uninstall faiss-cpu && pip install faiss-gpu"
        )


from .config import (
    FAISS_DIR,
    FAISS_LOCAL_DIR,
    OLLAMA_EMBEDDING_DIMENSION,
    OPENAI_EMBEDDING_DIMENSION,
    ensure_data_dir,
    get_gpu_free_memory_mb,
)


@dataclass
class VectorMetadata:
    """Metadata for a single vector in the index."""

    id: str  # section_id or summary_id
    video_id: str
    channel_id: str | None
    type: str  # 'section' or 'summary'


@dataclass
class VectorSearchResult:
    """Result from a vector search (low-level FAISS result)."""

    id: str
    video_id: str
    channel_id: str | None
    score: float
    type: str


class VectorStore:
    """FAISS-based vector store with metadata."""

    _gpu_check_done = False  # Class-level flag to run GPU check only once per process
    _gpu_resources: "faiss.StandardGpuResources | None" = None  # Shared GPU resources

    def __init__(
        self,
        name: str = "sections",
        dimension: int | None = None,
        index_dir: Path | None = None,
        use_local: bool = True,
        use_gpu: bool = True,
    ):
        """Initialize vector store.

        Args:
            name: Name of the index (e.g., 'sections', 'summaries')
            dimension: Vector dimension (auto-detected from use_local if not set)
            index_dir: Directory to store index files (auto-set from use_local if not set)
            use_local: If True, use local Ollama embeddings; if False, use OpenAI
            use_gpu: If True and GPU available, use GPU for search
        """
        # Run GPU upgrade check once per process on first VectorStore instantiation
        if not VectorStore._gpu_check_done:
            VectorStore._gpu_check_done = True
            check_gpu_upgrade_prompt()

        self.name = name
        self.use_local = use_local
        self.use_gpu = use_gpu and FAISS_GPU_AVAILABLE

        # Initialize shared GPU resources once, but only if enough memory available
        # StandardGpuResources allocates ~1GB temp buffer, plus we need room for index
        if self.use_gpu and VectorStore._gpu_resources is None:
            if _check_gpu_memory_available(min_free_mb=1500):
                try:
                    VectorStore._gpu_resources = faiss.StandardGpuResources()
                    logger.info("FAISS GPU resources initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU resources: {e}")
                    self.use_gpu = False
            else:
                logger.info("Insufficient GPU memory (<1.5GB free), using CPU for FAISS")
                self.use_gpu = False

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
        self._gpu_index: "faiss.GpuIndexFlatIP | None" = None
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

    def _get_search_index(self):
        """Get the appropriate index for searching (GPU if available, else CPU)."""
        if self._index is None:
            return None

        # Use GPU index if available
        if self.use_gpu and self._gpu_index is not None:
            return self._gpu_index

        # Create GPU index on first search if GPU enabled and enough memory
        if self.use_gpu and VectorStore._gpu_resources is not None and self._gpu_index is None:
            # Estimate memory needed: ~4 bytes per float32 * dimension * num_vectors + overhead
            estimated_mb = (self._index.ntotal * self.dimension * 4) / (1024 * 1024) + 256
            if not _check_gpu_memory_available(min_free_mb=int(estimated_mb)):
                logger.info(f"Insufficient GPU memory for index ({estimated_mb:.0f}MB needed), using CPU")
                self.use_gpu = False
                return self._index

            try:
                self._gpu_index = faiss.index_cpu_to_gpu(
                    VectorStore._gpu_resources, 0, self._index
                )
                logger.info(f"Created GPU index for '{self.name}' ({self._index.ntotal} vectors)")
                return self._gpu_index
            except Exception as e:
                logger.warning(f"Failed to create GPU index: {e}, falling back to CPU")
                # Disable GPU globally after failure to prevent segfaults from corrupted state
                self.use_gpu = False
                VectorStore._gpu_resources = None  # Clear corrupted resources

        return self._index

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_video_id: str | None = None,
        filter_channel_id: str | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_video_id: Only return results from this video
            filter_channel_id: Only return results from this channel

        Returns:
            List of VectorSearchResult sorted by score descending
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Get search index (GPU or CPU)
        search_index = self._get_search_index()

        # Normalize query
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # Over-fetch if filtering, then filter down
        # GPU has a max k limit of 2048, so cap it
        max_k = 2048 if self.use_gpu else self._index.ntotal
        fetch_k = top_k
        if filter_video_id or filter_channel_id:
            fetch_k = min(top_k * 10, self._index.ntotal, max_k)
        else:
            fetch_k = min(fetch_k, max_k)

        scores, indices = search_index.search(query, fetch_k)

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
                VectorSearchResult(
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
        self._gpu_index = None
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
