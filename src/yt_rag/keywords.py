"""Keyword extraction and synonym generation for yt-rag."""

import re
from collections import Counter
from dataclasses import dataclass

from .db import Database

# Common English stopwords to filter out
STOPWORDS = frozenset([
    # Articles, pronouns, prepositions
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them",
    "his", "her", "its", "their", "who", "what", "which", "whom", "whose",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into",
    "about", "after", "before", "between", "under", "over", "through",
    # Common verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "must", "shall", "get", "got", "getting", "go", "going", "went",
    "come", "coming", "came", "take", "taking", "took", "make", "making", "made",
    "see", "seeing", "saw", "know", "knowing", "knew", "think", "thinking", "thought",
    "want", "wanting", "wanted", "use", "using", "used", "find", "finding", "found",
    "give", "giving", "gave", "tell", "telling", "told", "say", "saying", "said",
    "put", "putting", "let", "keep", "keeping", "kept",
    # Conjunctions and connectors
    "and", "or", "but", "if", "then", "than", "so", "because", "when", "where",
    "while", "although", "however", "also", "just", "only", "even", "still",
    # Common adjectives/adverbs
    "good", "bad", "great", "nice", "new", "old", "big", "small", "little",
    "much", "many", "more", "most", "less", "very", "really", "actually",
    "pretty", "quite", "too", "well", "now", "here", "there", "back", "out",
    "up", "down", "off", "away", "again", "always", "never", "already",
    # Filler words common in transcripts
    "um", "uh", "like", "right", "okay", "yeah", "yes", "no", "oh", "ah",
    "gonna", "wanna", "gotta", "kinda", "sorta", "thing", "things", "stuff",
    "lot", "lots", "bit", "way", "kind", "type", "sort",
    # Numbers as words
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "last", "next",
    # Other common words
    "people", "time", "day", "year", "guy", "guys", "man", "men", "woman", "women",
    "person", "something", "anything", "everything", "nothing", "someone", "anyone",
    "everyone", "other", "another", "same", "different", "own", "every", "each",
    "all", "some", "any", "both", "few", "enough", "whole",
    # Contractions without apostrophe (from transcripts)
    "don", "didn", "doesn", "wasn", "weren", "won", "wouldn", "couldn", "shouldn",
    "haven", "hasn", "hadn", "isn", "aren", "ain",
    # More filler words
    "not", "how", "why", "mean", "doing", "look", "looks", "looking",
    "show", "maybe", "probably", "sure", "cuz", "gonna", "cause",
    "talk", "talking", "talked", "says", "need", "needs", "work", "working",
    "start", "started", "starting", "done", "goes", "feel", "feels", "feeling",
    "try", "trying", "tried", "call", "called", "hear", "heard",
    "believe", "guess", "suppose", "happen", "happened", "happening",
    "seem", "seems", "seemed", "matter", "mean", "means", "meant",
    "run", "running", "ran", "turn", "turning", "turned", "play", "playing",
    "move", "moving", "moved", "hold", "holding", "held", "leave", "leaving", "left",
    "bring", "bringing", "brought", "sit", "sitting", "sat", "stand", "standing", "stood",
    "set", "pull", "pulling", "pulled", "push", "pushing", "pushed",
    "point", "points", "end", "far", "though", "around",
    # Music marker common in transcripts
    "music",
])

# Minimum word length to consider
MIN_WORD_LENGTH = 3

# Minimum frequency to be considered a keyword
MIN_FREQUENCY = 5


@dataclass
class KeywordStats:
    """Statistics for a keyword."""
    keyword: str
    frequency: int
    video_count: int
    sample_contexts: list[str]  # Sample sentences containing the keyword


def extract_words(text: str) -> list[str]:
    """Extract words from text, filtering stopwords and short words."""
    # Extract alphanumeric words
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text.lower())

    # Filter stopwords and short words
    return [
        w for w in words
        if len(w) >= MIN_WORD_LENGTH and w not in STOPWORDS
    ]


def extract_keywords_from_video(
    db: Database,
    video_id: str,
    min_frequency: int = 2,
) -> Counter[str]:
    """Extract keywords from a single video's sections.

    Returns a Counter of keyword -> frequency.
    """
    sections = db.get_sections(video_id)
    if not sections:
        return Counter()

    word_counts: Counter[str] = Counter()

    for section in sections:
        words = extract_words(section.content)
        word_counts.update(words)

    # Filter by minimum frequency within this video
    return Counter({w: c for w, c in word_counts.items() if c >= min_frequency})


def extract_keywords_from_videos(
    db: Database,
    video_ids: list[str] | None = None,
    limit: int | None = None,
    min_video_frequency: int = 2,
    min_total_frequency: int = 5,
) -> list[KeywordStats]:
    """Extract keywords from multiple videos.

    Args:
        db: Database instance
        video_ids: Specific videos to analyze (None = all videos)
        limit: Max videos to process
        min_video_frequency: Min occurrences in a single video to count
        min_total_frequency: Min total occurrences across all videos

    Returns:
        List of KeywordStats sorted by frequency
    """
    conn = db.connect()

    # Get videos to process
    if video_ids is None:
        rows = conn.execute(
            "SELECT id FROM videos WHERE transcript_status = 'fetched' LIMIT ?",
            (limit or 999999,)
        ).fetchall()
        video_ids = [row[0] for row in rows]
    elif limit:
        video_ids = video_ids[:limit]

    if not video_ids:
        return []

    # Batch fetch ALL sections for these videos in ONE query
    placeholders = ",".join("?" * len(video_ids))
    rows = conn.execute(
        f"SELECT video_id, content FROM sections WHERE video_id IN ({placeholders})",
        video_ids,
    ).fetchall()

    # Group sections by video_id
    sections_by_video: dict[str, list[str]] = {}
    for video_id, content in rows:
        if video_id not in sections_by_video:
            sections_by_video[video_id] = []
        sections_by_video[video_id].append(content)

    # Track global counts
    global_counts: Counter[str] = Counter()
    video_presence: Counter[str] = Counter()  # How many videos contain each word
    contexts: dict[str, list[str]] = {}  # Sample contexts for each keyword

    for video_id in video_ids:
        contents = sections_by_video.get(video_id, [])
        if not contents:
            continue

        # Extract keywords from all sections of this video
        word_counts: Counter[str] = Counter()
        for content in contents:
            words = extract_words(content)
            word_counts.update(words)

        # Filter by minimum frequency within this video
        video_keywords = {w: c for w, c in word_counts.items() if c >= min_video_frequency}

        # Update global counts
        global_counts.update(video_keywords)

        # Track which videos contain each word
        for word in video_keywords:
            video_presence[word] += 1

            # Store sample context if we don't have enough
            if word not in contexts:
                contexts[word] = []
            if len(contexts[word]) < 3:
                # Find a section containing this word (use already-fetched content)
                for content in contents:
                    if word in content.lower():
                        context = _extract_context(content, word)
                        if context:
                            contexts[word].append(context)
                            break

    # Filter and build results
    results = []
    for word, freq in global_counts.most_common():
        if freq >= min_total_frequency:
            results.append(KeywordStats(
                keyword=word,
                frequency=freq,
                video_count=video_presence[word],
                sample_contexts=contexts.get(word, []),
            ))

    return results


def _extract_context(text: str, word: str, context_chars: int = 80) -> str:
    """Extract a short context around a word in text."""
    text_lower = text.lower()
    pos = text_lower.find(word)
    if pos == -1:
        return ""

    # Find start and end positions
    start = max(0, pos - context_chars // 2)
    end = min(len(text), pos + len(word) + context_chars // 2)

    # Adjust to word boundaries
    if start > 0:
        space = text.find(" ", start)
        if space != -1 and space < pos:
            start = space + 1
    if end < len(text):
        space = text.rfind(" ", pos, end)
        if space != -1:
            end = space

    context = text[start:end].strip()
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."

    return context


def analyze_videos(
    db: Database,
    video_ids: list[str],
    save_to_db: bool = False,
) -> list[KeywordStats]:
    """Analyze videos and optionally save keywords to database.

    Args:
        db: Database instance
        video_ids: Videos to analyze
        save_to_db: Whether to save keywords to the keywords table

    Returns:
        List of extracted keywords with stats
    """
    keywords = extract_keywords_from_videos(db, video_ids)

    if save_to_db:
        for kw in keywords:
            db.upsert_keyword(kw.keyword, kw.frequency, kw.video_count)

    return keywords


def suggest_synonyms_for_keyword(keyword: str) -> list[str]:
    """Suggest potential synonyms for a keyword.

    This is a simple heuristic-based approach. For better results,
    use an LLM or WordNet.
    """
    # Common automotive synonym patterns
    PATTERNS = {
        # Fuel-related
        "mpg": ["fuel economy", "fuel efficiency", "mileage", "gas mileage"],
        "fuel": ["gas", "petrol", "diesel"],
        "economy": ["efficiency", "mileage"],
        "efficiency": ["economy", "performance"],

        # Power-related
        "horsepower": ["hp", "power", "performance"],
        "hp": ["horsepower", "power"],
        "torque": ["power", "twist", "lb-ft", "nm"],
        "power": ["performance", "output"],
        "engine": ["motor", "powerplant", "powertrain"],
        "motor": ["engine", "powerplant"],

        # Transmission-related
        "transmission": ["gearbox", "trans", "shifter"],
        "automatic": ["auto", "autobox"],
        "manual": ["stick", "standard", "stick shift"],

        # Body-related
        "interior": ["cabin", "inside"],
        "exterior": ["outside", "body", "styling"],
        "dashboard": ["dash", "instrument panel"],
        "steering": ["wheel", "handling"],

        # Performance-related
        "acceleration": ["speed", "0-60", "launch"],
        "braking": ["brakes", "stopping"],
        "handling": ["cornering", "steering", "dynamics"],
        "suspension": ["ride", "dampers", "shocks"],

        # Quality-related
        "reliability": ["dependable", "durable", "quality"],
        "quality": ["build", "craftsmanship", "fit and finish"],

        # Price-related
        "price": ["cost", "msrp", "value"],
        "expensive": ["costly", "pricey", "premium"],
        "cheap": ["affordable", "budget", "value"],
        "value": ["worth", "bang for buck"],
    }

    return PATTERNS.get(keyword.lower(), [])


def generate_synonyms_with_llm(
    keywords: list[str],
    category: str = "general",
    use_local: bool = True,
) -> dict[str, list[str]]:
    """Generate synonyms using an LLM.

    Args:
        keywords: List of keywords to generate synonyms for
        category: Channel category (e.g., 'automotive', 'finance')
        use_local: Use Ollama (True) or OpenAI (False)

    Returns:
        Dict mapping keyword -> list of synonyms
    """
    from .openai_client import chat_completion, ollama_chat_completion

    if not keywords:
        return {}

    # Build prompt with category context
    keywords_str = ", ".join(keywords[:50])  # Limit to 50 keywords per batch

    system_prompt = f"""You are a synonym generator for a {category} video search system.
Generate synonyms, abbreviations, and related terms that users might search for.
Focus on domain-specific terminology for {category} content."""

    user_prompt = f"""For each keyword below, provide synonyms and related search terms.
Return ONLY a JSON object mapping each keyword to an array of synonyms.
Include abbreviations, alternate spellings, and closely related terms.
Skip keywords that have no useful synonyms.

Keywords: {keywords_str}

Example output format:
{{"mpg": ["fuel economy", "gas mileage", "fuel efficiency"], "hp": ["horsepower", "power"]}}

Return ONLY valid JSON, no explanation:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        if use_local:
            result = ollama_chat_completion(messages, temperature=0.3)
        else:
            result = chat_completion(messages, temperature=0.3)

        # Parse JSON response
        import json
        import re

        content = result.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in content:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)

        synonyms = json.loads(content)

        # Validate and clean response
        cleaned: dict[str, list[str]] = {}
        for kw, syns in synonyms.items():
            if isinstance(syns, list) and all(isinstance(s, str) for s in syns):
                # Filter out empty strings and the keyword itself
                cleaned_syns = [s.lower().strip() for s in syns if s.strip() and s.lower() != kw.lower()]
                if cleaned_syns:
                    cleaned[kw.lower()] = cleaned_syns

        return cleaned

    except Exception as e:
        # Fall back to heuristic approach on error
        import sys
        print(f"LLM synonym generation failed: {e}", file=sys.stderr)
        result = {}
        for kw in keywords:
            synonyms = suggest_synonyms_for_keyword(kw)
            if synonyms:
                result[kw] = synonyms
        return result


@dataclass
class SynonymRefreshResult:
    """Result of synonym refresh operation."""

    videos_analyzed: int = 0
    keywords_extracted: int = 0
    synonyms_added: int = 0
    channels_processed: list[str] | None = None


def refresh_synonyms(
    db: Database,
    channel_id: str | None = None,
    force: bool = False,
    min_frequency: int = 3,
    top_keywords: int = 100,
    use_llm: bool = True,
    use_local: bool = True,
) -> SynonymRefreshResult:
    """Refresh synonyms for videos, extracting keywords and generating synonym suggestions.

    Processes each channel separately to get channel-specific keywords,
    then uses the channel's category for domain-specific synonym generation.

    Args:
        db: Database instance
        channel_id: Specific channel to process (None = all channels)
        force: If True, regenerate even if synonyms exist
        min_frequency: Minimum keyword frequency to consider (per channel)
        top_keywords: Number of top keywords per channel to generate synonyms for
        use_llm: Use LLM for synonym generation (True) or heuristics only (False)
        use_local: Use Ollama (True) or OpenAI (False) for LLM

    Returns:
        SynonymRefreshResult with counts
    """
    result = SynonymRefreshResult()
    conn = db.connect()

    # Get channels to process
    if channel_id:
        channel_rows = conn.execute(
            "SELECT id, name, category FROM channels WHERE id = ?",
            (channel_id,),
        ).fetchall()
    else:
        channel_rows = conn.execute(
            "SELECT id, name, category FROM channels"
        ).fetchall()

    if not channel_rows:
        return result

    result.channels_processed = []

    # Process each channel separately
    for ch_id, ch_name, ch_category in channel_rows:
        category = ch_category or "general"

        # Get videos with sections for this channel
        video_rows = conn.execute(
            """
            SELECT DISTINCT v.id FROM videos v
            JOIN sections s ON v.id = s.video_id
            WHERE v.channel_id = ? AND v.transcript_status = 'fetched'
            """,
            (ch_id,),
        ).fetchall()

        if not video_rows:
            continue

        video_ids = [r[0] for r in video_rows]
        result.videos_analyzed += len(video_ids)
        result.channels_processed.append(ch_name)

        # Extract top keywords for this channel
        keywords = extract_keywords_from_videos(
            db,
            video_ids,
            min_total_frequency=min_frequency,
        )

        if not keywords:
            continue

        # Save keywords to database
        for kw in keywords:
            db.upsert_keyword(kw.keyword, kw.frequency, kw.video_count)

        result.keywords_extracted += len(keywords)

        # Generate synonyms for top keywords of this channel
        top_kws = keywords[:top_keywords]
        top_kw_names = [kw.keyword for kw in top_kws]

        if use_llm:
            # Use LLM to generate synonyms in batches
            batch_size = 50
            channel_synonyms: dict[str, list[str]] = {}

            for i in range(0, len(top_kw_names), batch_size):
                batch = top_kw_names[i : i + batch_size]
                batch_synonyms = generate_synonyms_with_llm(
                    batch, category=category, use_local=use_local
                )
                channel_synonyms.update(batch_synonyms)

            # Save LLM-generated synonyms
            for kw, syns in channel_synonyms.items():
                for syn in syns:
                    existing = conn.execute(
                        "SELECT 1 FROM synonyms WHERE keyword = ? AND synonym = ?",
                        (kw.lower(), syn.lower()),
                    ).fetchone()

                    if not existing or force:
                        db.add_synonym(kw, syn, source="llm", approved=True)
                        result.synonyms_added += 1
        else:
            # Use heuristic approach
            for kw in top_kws:
                synonyms = suggest_synonyms_for_keyword(kw.keyword)
                if synonyms:
                    for syn in synonyms:
                        existing = conn.execute(
                            "SELECT 1 FROM synonyms WHERE keyword = ? AND synonym = ?",
                            (kw.keyword.lower(), syn.lower()),
                        ).fetchone()

                        if not existing or force:
                            db.add_synonym(kw.keyword, syn, source="heuristic", approved=False)
                            result.synonyms_added += 1

    return result
