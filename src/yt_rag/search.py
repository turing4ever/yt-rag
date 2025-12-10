"""RAG search functionality."""

import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_EMBED_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_QUERY_MODEL,
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
from .openai_client import chat_completion, embed_text, ollama_chat_completion, ollama_embed_text
from .vectorstore import get_sections_store


class QueryType(Enum):
    """Classification of query intent for retrieval strategy."""

    ENTITY = "entity"  # Looking for specific car model, brand, etc.
    TOPIC = "topic"  # General topic like "fuel economy", "reliability"
    COMPARISON = "comparison"  # Comparing entities: "X vs Y"
    LIST = "list"  # "How many videos about X", "which cars have..."
    FACTUAL = "factual"  # General question
    FOLLOWUP = "followup"  # Follow-up question referencing previous results
    POPULARITY = "popularity"  # Asking about most popular/viewed/liked videos
    META = "meta"  # Library stats: "how many videos", "what channels"


@dataclass
class RelevanceMetrics:
    """Metrics for evaluating search result quality."""

    semantic_score: float  # Raw FAISS cosine similarity
    keyword_match_count: int  # How many query terms found
    synonym_match_count: int  # How many synonyms found
    title_match: bool  # Query term in video/section title
    exact_phrase_match: bool  # Exact query phrase found
    relevance_multiplier: float  # Multiplier based on keyword matches
    final_score: float  # Combined final score

    @property
    def confidence(self) -> str:
        """Return confidence level based on metrics."""
        if self.title_match or self.keyword_match_count >= 2:
            return "high"
        elif self.keyword_match_count >= 1 or self.synonym_match_count >= 1:
            return "medium"
        return "low"


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
    metrics: RelevanceMetrics | None = None  # Detailed relevance metrics
    view_count: int | None = None  # Video view count
    like_count: int | None = None  # Video like count
    published_at: datetime | None = None  # Video publish date


@dataclass
class SearchResponse:
    """Complete search response."""

    query: str
    hits: list[SearchHit]
    answer: str | None
    tokens_embedding: int
    tokens_chat: int
    latency_ms: int


@dataclass
class QueryAnalysis:
    """LLM-analyzed query structure."""

    query_type: QueryType
    keywords: list[str]  # Normalized search keywords
    original_query: str
    reasoning: str | None = None  # LLM's reasoning (for debugging)
    time_filter_days: int | None = None  # For popularity queries: filter to videos published in last N days


# Prompt for LLM query analysis - optimized for speed (~150 tokens vs ~600)
QUERY_ANALYSIS_PROMPT = """\
Classify this video search query. Return JSON only.

Query: {query}

Types (check in this order):
1. meta: asks about library/system stats (how many videos/sections/channels, index info, GPU/CPU, embedding model)
2. followup: uses pronouns/ordinals referring to UNSPECIFIED items: "those", "them", "they", "it", "the first", "which one", "among"
3. comparison: EXPLICIT "X vs Y" or "X compared to Y" with two named items
4. popularity: asks about most viewed/liked VIDEOS (must mention video/views)
5. entity: ONLY a product/model name with no other words ("G550", "iPhone 15")
6. topic: everything else (questions, descriptions, general subjects)

Return: {{"type": "...", "keywords": [...], "time_filter_days": null}}

Keywords: extract search terms, normalize compound words, keep model numbers intact.
time_filter_days: for popularity only - "this week"=7, "this month"=30, "this year"=365."""


def analyze_query_with_llm(
    query: str,
    use_local: bool = True,
    model: str | None = None,
) -> QueryAnalysis | None:
    """Use LLM to analyze query intent and extract keywords.

    Args:
        query: The search query
        use_local: Use Ollama (True) or OpenAI (False)
        model: Model to use (defaults based on use_local)

    Returns:
        QueryAnalysis or None if parsing fails
    """
    import json
    import logging

    logger = logging.getLogger(__name__)

    prompt = QUERY_ANALYSIS_PROMPT.format(query=query)

    try:
        if use_local:
            model = model or DEFAULT_OLLAMA_QUERY_MODEL  # Use fast 3b model for parsing
            response = ollama_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,  # Deterministic for consistency
            )
        else:
            model = model or DEFAULT_CHAT_MODEL
            response = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
                max_tokens=200,
            )

        content = response.content.strip()

        # Try to extract JSON from response
        # Handle cases where LLM wraps in ```json blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)

        # Parse query type (handle both "type" and "query_type" for compatibility)
        query_type_str = data.get("type") or data.get("query_type", "factual")
        query_type_str = query_type_str.lower()
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.FACTUAL

        keywords = [k.lower() for k in data.get("keywords", [])]
        reasoning = data.get("reasoning")
        time_filter_days = data.get("time_filter_days")

        # Post-process: validate popularity classification
        # True popularity queries ask about VIDEO metrics (views, likes)
        # Queries about "most popular [subject]" should be topic queries
        if query_type == QueryType.POPULARITY:
            query_lower = query.lower()
            # Check if query explicitly mentions video/views
            video_terms = ["video", "videos", "view", "views", "liked", "likes", "watched"]
            has_video_term = any(term in query_lower for term in video_terms)
            # If no video term, it's probably a topic query about a popular subject
            if not has_video_term:
                logger.debug(
                    f"Reclassifying popularity -> topic: no video terms in query"
                )
                query_type = QueryType.TOPIC
                # Extract keywords from query if LLM didn't provide them
                if not keywords:
                    # Simple extraction: remove common words and use remaining terms
                    stopwords = {"what", "what's", "whats", "is", "the", "most", "in", "a", "an", "are", "how", "many", "which"}
                    words = re.findall(r'\b\w+\b', query_lower)
                    keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Validate time_filter_days is a positive integer
        if time_filter_days is not None:
            try:
                time_filter_days = int(time_filter_days)
                if time_filter_days <= 0:
                    time_filter_days = None
            except (ValueError, TypeError):
                time_filter_days = None

        result = QueryAnalysis(
            query_type=query_type,
            keywords=keywords,
            original_query=query,
            reasoning=reasoning,
            time_filter_days=time_filter_days,
        )

        # Log for debugging
        logger.debug(
            f"LLM Query Analysis: query={query!r} -> "
            f"type={query_type.value}, keywords={keywords}, reasoning={reasoning!r}"
        )

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM query analysis failed: {e}")
        return None


RAG_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on YouTube video transcripts.

CRITICAL: You must ONLY answer based on the provided context. NEVER mention videos, channels, or content that is not explicitly in the context. If the context doesn't contain relevant information, say so - do not guess or make up answers.

IMPORTANT: Write a short paragraph answer only. Do NOT:
- List numbered items
- Include URLs or timestamps
- Format as "Channel | Video Title"
- Repeat the source entries
- Mention any videos not in the provided context

Only discuss topics explicitly mentioned in the user's query. If the user searches a single keyword (like a car model), just say what videos are available - do not describe the video content or add any details.

The system displays sources separately. Keep answers to 1-2 sentences."""

RAG_USER_PROMPT = """\
Context from video transcripts (found via semantic search and synonym matching):

{context}

Question: {question}"""


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

    # Get synonyms and expand terms
    synonyms = get_synonyms_map()
    expanded_terms = set(terms)
    for term in terms:
        if term in synonyms:
            expanded_terms.update(synonyms[term])

    # Normalize texts for case-insensitive matching
    video_title_lower = video_title.lower()
    section_title_lower = section_title.lower()
    section_content_lower = section_content.lower()

    boost = 0.0

    for term in expanded_terms:
        # Check each level - only count once per term per level
        if term in video_title_lower:
            boost += KEYWORD_BOOST_VIDEO_TITLE
        if term in section_title_lower:
            boost += KEYWORD_BOOST_SECTION_TITLE
        if term in section_content_lower:
            boost += KEYWORD_BOOST_SECTION_CONTENT

    return boost


# Cache for synonyms (refreshed on first use)
_synonyms_cache: dict[str, list[str]] | None = None


# Default synonyms (used if database is empty or unavailable)
DEFAULT_SYNONYMS = {
    "mpg": ["fuel", "efficiency", "economy", "mileage", "consumption"],
    "fuel": ["mpg", "efficiency", "economy", "mileage", "gas"],
    "hp": ["horsepower", "power", "engine"],
    "horsepower": ["hp", "power", "engine"],
    "torque": ["power", "engine", "lb-ft"],
    "price": ["cost", "msrp", "expensive", "cheap", "value"],
    "cost": ["price", "msrp", "expensive", "value"],
    "interior": ["cabin", "inside", "seats", "dashboard"],
    "exterior": ["outside", "body", "styling", "design"],
    "reliability": ["reliable", "dependable", "issues", "problems"],
}


def get_synonyms_map(refresh: bool = False) -> dict[str, list[str]]:
    """Get synonyms map from database, with fallback to defaults.

    Uses a module-level cache for performance.
    """
    global _synonyms_cache

    if _synonyms_cache is not None and not refresh:
        return _synonyms_cache

    try:
        db = Database()
        db.init()
        db_synonyms = db.get_all_synonyms(approved_only=True)
        db.close()

        if db_synonyms:
            _synonyms_cache = db_synonyms
        else:
            # Use defaults if database is empty
            _synonyms_cache = DEFAULT_SYNONYMS.copy()
    except Exception:
        # Fallback to defaults on any database error
        _synonyms_cache = DEFAULT_SYNONYMS.copy()

    return _synonyms_cache


def refresh_synonyms_cache() -> None:
    """Force refresh of synonyms cache from database."""
    get_synonyms_map(refresh=True)


# Patterns for entity detection (alphanumeric model names like G550, Model 3, etc.)
ENTITY_PATTERN = re.compile(
    r"\b(?:"
    r"[A-Z]?\d{2,4}[A-Z]?"  # G550, 911, M3, RS7
    r"|[A-Z]{1,3}\d{1,4}"  # GT500, S550, M5, RS6
    r"|Model\s*[3SXY]"  # Tesla Model 3, S, X, Y
    r"|[A-Z]-?Class"  # Mercedes S-Class, G-Class
    r"|[A-Z]{2,}\s*\d+"  # BMW M5, AMG GT
    r")\b",
    re.IGNORECASE,
)

# Comparison patterns
COMPARISON_PATTERN = re.compile(
    r"\b(?:vs\.?|versus|compared?\s+to|or)\b",
    re.IGNORECASE,
)

# List/count patterns
LIST_PATTERN = re.compile(
    r"\b(?:how\s+many|which\s+(?:cars?|videos?)|list\s+(?:of|all)|what\s+(?:cars?|videos?))\b",
    re.IGNORECASE,
)

# Topic keywords (not entity-specific)
TOPIC_KEYWORDS = frozenset([
    "mpg", "fuel", "economy", "efficiency", "mileage",
    "reliability", "reliable", "problems", "issues",
    "interior", "exterior", "cabin", "design",
    "price", "cost", "value", "expensive", "cheap",
    "performance", "speed", "acceleration", "handling",
    "comfort", "luxury", "quality", "review",
    "towing", "payload", "capacity", "cargo",
    "safety", "crash", "rating",
])


# Patterns for META queries (library stats without specific topic)
META_PATTERNS = [
    r"^how many videos\??$",  # exactly "how many videos?"
    r"^how many videos (are|do you have|are there|in total)\??$",
    r"^how many (channels|sections|segments|summaries)\??$",
    r"^how many (channels|videos)( and (channels|videos))?\??$",  # combined queries
    r"^what channels\??$",
    r"^list (all )?channels\??$",
    r"^show (all )?channels\??$",
    r"^what('s| is) (in )?(the |my )?library\??$",
    r"^library stats?\??$",
    r"^stats?\??$",
    r"^(system |index )?(info|status|stats?)\??$",
    r"^(is it |are you )?(using )?(gpu|cpu)\??$",
    r"^what (embedding|model)( are you using)?\??$",
]
META_PATTERN = re.compile("|".join(META_PATTERNS), re.IGNORECASE)


def classify_query(query: str) -> QueryType:
    """Classify query intent for retrieval strategy selection.

    Args:
        query: The search query

    Returns:
        QueryType indicating the query intent
    """
    query_lower = query.lower()
    query_terms = set(re.findall(r"\b[a-zA-Z0-9]+\b", query_lower))

    # Check for META queries (library stats) - before LIST to avoid overlap
    if META_PATTERN.search(query):
        return QueryType.META

    # Check for list/count patterns first
    if LIST_PATTERN.search(query):
        return QueryType.LIST

    # Check for comparison patterns
    if COMPARISON_PATTERN.search(query):
        return QueryType.COMPARISON

    # Check for entity patterns (model names)
    has_entity = bool(ENTITY_PATTERN.search(query))

    # Check for topic keywords
    has_topic = bool(query_terms & TOPIC_KEYWORDS)

    # If only entity, it's an entity search
    if has_entity and not has_topic:
        return QueryType.ENTITY

    # If entity + topic, still entity-focused but with topic filter
    if has_entity and has_topic:
        return QueryType.ENTITY

    # Pure topic search
    if has_topic:
        return QueryType.TOPIC

    # Short alphanumeric queries are likely entity searches
    # e.g., "G550", "Tacoma", "Bronco"
    if len(query_terms) <= 2:
        # Check if any term looks entity-like (capitalized, alphanumeric mix)
        for term in query.split():
            if re.match(r"^[A-Z][a-z]*\d*$|^\d+[A-Z]*$|^[A-Z]{2,}\d*$", term):
                return QueryType.ENTITY

    return QueryType.FACTUAL


def extract_entity_terms(query: str) -> set[str]:
    """Extract entity terms from query (model names, brand names).

    Args:
        query: The search query

    Returns:
        Set of entity terms (lowercase)
    """
    entities = set()

    # Find alphanumeric model patterns
    for match in ENTITY_PATTERN.finditer(query):
        entities.add(match.group().lower().replace(" ", ""))

    # Also include capitalized words that might be model/brand names
    for word in query.split():
        # Alphanumeric mix (G550, RS7) or capitalized brand (Tacoma, Bronco)
        if re.match(r"^[A-Z]?[a-z]*\d+[A-Z]?$|^[A-Z][a-z]{3,}$", word):
            entities.add(word.lower())

    return entities


def expand_terms_with_synonyms(terms: set[str]) -> set[str]:
    """Expand terms with their synonyms.

    Args:
        terms: Set of terms to expand

    Returns:
        Expanded set including synonyms
    """
    synonyms = get_synonyms_map()
    expanded = set(terms)

    for term in terms:
        if term in synonyms:
            expanded.update(synonyms[term])

    return expanded


def compute_relevance_metrics(
    query: str,
    query_type: QueryType,
    semantic_score: float,
    video_title: str,
    section_title: str,
    section_content: str,
) -> RelevanceMetrics:
    """Compute detailed relevance metrics for a search result.

    Uses multiplicative scoring instead of additive boost.
    For entity searches, penalizes results without keyword matches.

    Args:
        query: The search query
        query_type: Classified query type
        semantic_score: Raw FAISS similarity score
        video_title: Video title
        section_title: Section title
        section_content: Section content

    Returns:
        RelevanceMetrics with detailed scoring
    """
    # Extract and expand query terms
    query_terms = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", query.lower()))
    synonyms = get_synonyms_map()

    # Build expanded term set with synonyms
    expanded_terms = set(query_terms)
    for term in query_terms:
        if term in synonyms:
            expanded_terms.update(synonyms[term])

    # Normalize texts
    video_title_lower = video_title.lower()
    section_title_lower = section_title.lower()
    section_content_lower = section_content.lower()
    all_text = f"{video_title_lower} {section_title_lower} {section_content_lower}"

    # Count matches
    keyword_matches = 0
    synonym_matches = 0
    title_match = False

    for term in query_terms:
        if term in video_title_lower or term in section_title_lower:
            keyword_matches += 1
            title_match = True
        elif term in section_content_lower:
            keyword_matches += 1

    for term in expanded_terms - query_terms:  # Only synonyms
        if term in all_text:
            synonym_matches += 1

    # Check for exact phrase match
    exact_phrase_match = query.lower() in all_text

    # Compute relevance multiplier based on query type and matches
    if query_type == QueryType.ENTITY:
        # Entity searches: strongly penalize no keyword match
        if keyword_matches == 0 and synonym_matches == 0:
            relevance_multiplier = 0.3  # Heavy penalty
        elif keyword_matches == 0 and synonym_matches > 0:
            relevance_multiplier = 0.7  # Moderate penalty (synonym only)
        elif title_match:
            relevance_multiplier = 1.5  # Boost for title match
        else:
            relevance_multiplier = 1.2  # Slight boost for content match
    elif query_type == QueryType.TOPIC:
        # Topic searches: mild penalty for no match
        if keyword_matches == 0 and synonym_matches == 0:
            relevance_multiplier = 0.6
        elif keyword_matches > 0:
            relevance_multiplier = 1.0 + (0.1 * keyword_matches)
        else:
            relevance_multiplier = 0.9
    else:
        # Factual/list/comparison: balanced approach
        if title_match:
            relevance_multiplier = 1.3
        elif keyword_matches > 0:
            relevance_multiplier = 1.1
        elif synonym_matches > 0:
            relevance_multiplier = 1.0
        else:
            relevance_multiplier = 0.8

    # Also add the old additive boost for compatibility
    additive_boost = 0.0
    for term in expanded_terms:
        if term in video_title_lower:
            additive_boost += KEYWORD_BOOST_VIDEO_TITLE
        if term in section_title_lower:
            additive_boost += KEYWORD_BOOST_SECTION_TITLE
        if term in section_content_lower:
            additive_boost += KEYWORD_BOOST_SECTION_CONTENT

    # Combine: multiplicative relevance + additive boost
    final_score = (semantic_score * relevance_multiplier) + additive_boost

    return RelevanceMetrics(
        semantic_score=semantic_score,
        keyword_match_count=keyword_matches,
        synonym_match_count=synonym_matches,
        title_match=title_match,
        exact_phrase_match=exact_phrase_match,
        relevance_multiplier=relevance_multiplier,
        final_score=final_score,
    )


def filter_with_llm_analysis(
    query: str,
    candidates: list,
    db: Database,
    use_local: bool = True,
) -> tuple[list, QueryAnalysis | None]:
    """Filter candidates using LLM-analyzed keywords.

    Uses LLM to extract normalized keywords from the query, then filters
    candidates to those matching at least one keyword.

    Args:
        query: The search query
        candidates: List of FAISS search results
        db: Database for fetching section/video data
        use_local: Use Ollama (True) or OpenAI (False)

    Returns:
        Tuple of (filtered candidates, QueryAnalysis or None if LLM failed)
    """
    import logging

    logger = logging.getLogger(__name__)

    # Get LLM analysis
    analysis = analyze_query_with_llm(query, use_local=use_local)

    if not analysis:
        # LLM failed completely - return all candidates unfiltered
        logger.warning("LLM analysis failed, returning unfiltered candidates")
        return candidates, None

    # For follow-up queries, don't filter - the caller should use cached results
    if analysis.query_type == QueryType.FOLLOWUP:
        logger.debug(f"Follow-up query detected: {analysis.reasoning}")
        return candidates, analysis

    # If no keywords extracted, return all candidates
    if not analysis.keywords:
        logger.debug("No keywords extracted, returning unfiltered candidates")
        return candidates, analysis

    # Use LLM-extracted keywords
    keywords = set(analysis.keywords)
    synonyms_map = get_synonyms_map()

    # Expand keywords with synonyms
    expanded_keywords: set[str] = set()
    for kw in keywords:
        expanded_keywords.add(kw)
        if kw in synonyms_map:
            expanded_keywords.update(synonyms_map[kw])

    logger.debug(
        f"LLM filter: keywords={list(keywords)}, "
        f"expanded={list(expanded_keywords)}, type={analysis.query_type.value}"
    )

    # Determine min matches based on query type
    if analysis.query_type in (QueryType.ENTITY, QueryType.COMPARISON):
        # For entity/comparison, require all main keywords
        min_matches = len(keywords)
    else:
        # For topic/list/factual, require at least 1
        min_matches = 1

    filtered = []
    for result in candidates:
        section = db.get_section(result.id)
        if not section:
            continue

        video = db.get_video(section.video_id)
        if not video:
            continue

        all_text = f"{video.title} {section.title} {section.content}".lower()

        # Count keyword matches
        matches = 0
        for kw in keywords:
            # Check if keyword or any of its synonyms match
            kw_variants = {kw}
            if kw in synonyms_map:
                kw_variants.update(synonyms_map[kw])

            for variant in kw_variants:
                pattern = r"\b" + re.escape(variant) + r"\b"
                if re.search(pattern, all_text):
                    matches += 1
                    break

        if matches >= min_matches:
            filtered.append(result)

    logger.debug(f"LLM filter: {len(candidates)} -> {len(filtered)} candidates")

    return filtered, analysis


def find_precise_timestamp(
    query: str,
    section: "Section",
    db: "Database",
) -> float | None:
    """Find the precise timestamp within a section where query terms appear.

    Searches through the raw transcript segments within the section's time range
    to find where the query terms are first mentioned.

    Args:
        query: The search query
        section: The section to search within
        db: Database to fetch segments

    Returns:
        Precise timestamp in seconds, or None if no match found
    """
    # Extract meaningful terms from query
    terms = re.findall(r"\b[a-zA-Z0-9]{2,}\b", query.lower())
    if not terms:
        return None

    # Get segments for this video within the section's time range
    segments = db.get_segments(section.video_id)
    if not segments:
        return None

    # Filter to segments within section time range
    start_time = section.start_time or 0
    end_time = section.end_time or float("inf")

    section_segments = [
        s for s in segments if start_time <= s.start_time <= end_time
    ]

    # Find first segment that contains any query term
    for segment in section_segments:
        segment_text_lower = segment.text.lower()
        for term in terms:
            if term in segment_text_lower:
                return segment.start_time

    return None


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
        if hit.view_count:
            meta_parts.append(f"Views: {hit.view_count:,}")
        if hit.like_count:
            meta_parts.append(f"Likes: {hit.like_count:,}")
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
    import logging

    logger = logging.getLogger(__name__)
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

    # Run embedding and LLM query analysis in parallel for lower latency
    # Both are I/O bound (API calls to Ollama) so threading helps
    embed_result = None
    llm_analysis = None

    def do_embed():
        if use_local:
            return ollama_embed_text(query, embedding_model)
        else:
            return embed_text(query, embedding_model)

    def do_analyze():
        return analyze_query_with_llm(query, use_local=use_local)

    with ThreadPoolExecutor(max_workers=2) as executor:
        embed_future = executor.submit(do_embed)
        analyze_future = executor.submit(do_analyze)

        embed_result = embed_future.result()
        llm_analysis = analyze_future.result()

    tokens_embedding = embed_result.tokens_used

    # Use LLM analysis for query type, fall back to regex-based classification
    if llm_analysis:
        query_type = llm_analysis.query_type
        logger.info(
            f"LLM analysis: type={query_type.value}, "
            f"keywords={llm_analysis.keywords}, "
            f"reasoning={llm_analysis.reasoning!r}"
        )
    else:
        query_type = classify_query(query)

    # Search vector store with oversampling for re-ranking
    oversample_k = max(top_k, DEFAULT_TOP_K_OVERSAMPLE)
    results = store.search(
        query_embedding=embed_result.embedding,
        top_k=oversample_k,
        filter_video_id=video_id,
        filter_channel_id=channel_id,
    )

    # Filter by score threshold (use lower threshold for entity searches)
    min_threshold = score_threshold * 0.7 if query_type == QueryType.ENTITY else score_threshold
    results = [r for r in results if r.score >= min_threshold]

    # Apply LLM keyword filtering if analysis succeeded
    if llm_analysis and llm_analysis.keywords and llm_analysis.query_type != QueryType.FOLLOWUP:
        # Filter results using LLM-extracted keywords
        synonyms_map = get_synonyms_map()
        keywords = set(llm_analysis.keywords)

        # Expand keywords with synonyms
        expanded_keywords: set[str] = set()
        for kw in keywords:
            expanded_keywords.add(kw)
            if kw in synonyms_map:
                expanded_keywords.update(synonyms_map[kw])

        # Determine min matches based on query type
        if llm_analysis.query_type in (QueryType.ENTITY, QueryType.COMPARISON):
            min_matches = len(keywords)
        else:
            min_matches = 1

        filtered = []
        for result in results:
            section = db.get_section(result.id)
            if not section:
                continue
            video = db.get_video(section.video_id)
            if not video:
                continue

            all_text = f"{video.title} {section.title} {section.content}".lower()

            matches = 0
            for kw in keywords:
                kw_variants = {kw}
                if kw in synonyms_map:
                    kw_variants.update(synonyms_map[kw])
                for variant in kw_variants:
                    pattern = r"\b" + re.escape(variant) + r"\b"
                    if re.search(pattern, all_text):
                        matches += 1
                        break

            if matches >= min_matches:
                filtered.append(result)

        logger.debug(f"LLM filter: {len(results)} -> {len(filtered)} candidates")
        results = filtered

    # Handle popularity queries - search by view count instead of semantics
    if llm_analysis and llm_analysis.query_type == QueryType.POPULARITY:
        logger.info(f"Popularity query detected: {llm_analysis.reasoning}")

        # Extract channel filter from keywords
        channel_filter = None
        title_filter = None
        if llm_analysis.keywords:
            keyword_str = " ".join(llm_analysis.keywords)
            channels = db.list_channels()
            for ch in channels:
                if keyword_str.lower() in ch.name.lower():
                    channel_filter = ch.id
                    break
                if ch.handle and keyword_str.lower() in ch.handle.lower():
                    channel_filter = ch.id
                    break
            if not channel_filter:
                title_filter = keyword_str

        # Calculate published_after date from time_filter_days
        published_after = None
        if llm_analysis.time_filter_days:
            published_after = datetime.now() - timedelta(days=llm_analysis.time_filter_days)
            logger.info(f"Time filter: last {llm_analysis.time_filter_days} days (after {published_after.date()})")

        # Get top videos by view count
        top_videos = db.get_top_videos_by_views(
            limit=top_k,
            channel_id=channel_filter or channel_id,
            title_contains=title_filter,
            published_after=published_after,
        )

        # Build hits from top videos
        hits: list[SearchHit] = []
        for video in top_videos:
            sections = db.get_sections(video.id)
            if not sections:
                continue
            section = sections[0]
            channel = db.get_channel(video.channel_id) if video.channel_id else None
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
                score=1.0,
                timestamp_url=timestamp_url,
                view_count=video.view_count,
                like_count=video.like_count,
                published_at=video.published_at,
            )
            hits.append(hit)

        # Generate answer if requested
        answer = None
        tokens_chat = 0
        if generate_answer and hits:
            context = format_context(hits, db=db)
            messages = [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": RAG_USER_PROMPT.format(context=context, question=query)},
            ]
            if use_local:
                local_model = chat_model if not chat_model.startswith("gpt") else DEFAULT_OLLAMA_MODEL
                chat_result = ollama_chat_completion(messages=messages, model=local_model, temperature=temperature)
            else:
                chat_result = chat_completion(messages=messages, model=chat_model, temperature=temperature, max_tokens=max_tokens)
            answer = chat_result.content
            tokens_chat = chat_result.tokens_input + chat_result.tokens_output
        elif generate_answer and not hits:
            answer = "No videos with view count data found. Run 'yt-rag refresh-meta --video --force' to populate view counts."

        latency_ms = int((time.time() - start_time) * 1000)
        return SearchResponse(
            query=query,
            hits=hits,
            answer=answer,
            tokens_embedding=tokens_embedding,
            tokens_chat=tokens_chat,
            latency_ms=latency_ms,
        )

    # Build search hits with relevance-based scores
    candidate_hits: list[tuple[float, SearchHit]] = []
    for result in results:
        section = db.get_section(result.id)
        if not section:
            continue

        video = db.get_video(section.video_id)
        if not video:
            continue

        channel = db.get_channel(video.channel_id) if video.channel_id else None

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
        precise_ts = find_precise_timestamp(query, section, db)
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

        # Use local Ollama or OpenAI based on use_local flag
        if use_local:
            # Use default Ollama model if chat_model is an OpenAI model
            local_model = chat_model if not chat_model.startswith("gpt") else DEFAULT_OLLAMA_MODEL
            chat_result = ollama_chat_completion(
                messages=messages,
                model=local_model,
                temperature=temperature,
            )
        else:
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
