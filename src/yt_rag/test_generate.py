"""Test case generation from video transcripts.

This module provides tools to automatically generate benchmark test cases
by analyzing actual video content in the database.

Workflow:
1. prepare_raw_data(): Sample videos and save to raw_videos.json
2. analyze_videos(): Use LLM to extract entities/topics/facts -> video_analysis_{model}.json
3. build_tests(): Generate test queries from analysis -> benchmark_generated_{model}.json

Supports both local (Ollama) and remote (OpenAI) LLMs for analysis.
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_CHAT_MODEL, DEFAULT_OLLAMA_MODEL
from .db import Database
from .models import Section, Video
from .openai_client import chat_completion, ollama_chat_completion

logger = logging.getLogger(__name__)


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in filenames.

    Converts model names like "gpt-4o-mini" or "qwen2.5:7b-instruct" to
    filesystem-safe names like "gpt-4o-mini" or "qwen2.5_7b-instruct".

    Args:
        model: Raw model name

    Returns:
        Sanitized model name safe for filenames
    """
    # Replace colons with underscores (common in Ollama model names)
    safe = model.replace(":", "_")
    # Replace slashes with underscores (for paths like org/model)
    safe = safe.replace("/", "_")
    # Remove or replace other problematic characters
    safe = re.sub(r'[<>"|?*]', "", safe)
    # Collapse multiple underscores/dashes
    safe = re.sub(r"[-_]+", "-", safe)
    # Remove leading/trailing dashes
    safe = safe.strip("-")
    return safe


def get_analysis_filename(model: str) -> str:
    """Get the analysis output filename for a given model.

    Args:
        model: LLM model name

    Returns:
        Filename like "video_analysis_gpt-4o-mini.json"
    """
    safe_model = sanitize_model_name(model)
    return f"video_analysis_{safe_model}.json"


def get_benchmark_filename(model: str) -> str:
    """Get the benchmark output filename for a given model.

    Args:
        model: LLM model name (used during analysis)

    Returns:
        Filename like "benchmark_generated_gpt-4o-mini.json"
    """
    safe_model = sanitize_model_name(model)
    return f"benchmark_generated_{safe_model}.json"

# Output directory for test data files
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "data"


@dataclass
class RawVideoData:
    """Raw video data for analysis."""

    video_id: str
    title: str
    channel_id: str
    channel_name: str
    duration_seconds: int | None
    view_count: int | None
    sections: list[dict]  # [{id, title, content, start_time}]


@dataclass
class VideoAnalysis:
    """LLM-analyzed video content."""

    video_id: str
    title: str
    channel_name: str
    entities: list[str]  # Product names, model numbers, brands, people
    topics: list[str]  # Main themes discussed
    facts: list[dict]  # [{section_id, fact, keywords}]
    comparisons: list[dict]  # [{items: [a, b], aspect}]
    howtos: list[dict]  # [{action, subject, section_id}]


@dataclass
class PrepareResult:
    """Result of prepare_raw_data()."""

    channels_sampled: int
    videos_sampled: int
    output_file: Path


@dataclass
class AnalyzeResult:
    """Result of analyze_videos()."""

    videos_analyzed: int
    total_entities: int
    total_topics: int
    total_facts: int
    output_file: Path


@dataclass
class BuildResult:
    """Result of build_tests()."""

    tests_generated: int
    by_type: dict[str, int]
    output_file: Path


def prepare_raw_data(
    db: Database,
    videos_per_channel: int = 15,
    output_path: Path | None = None,
) -> PrepareResult:
    """Sample videos from database and save raw data to JSON.

    Args:
        db: Database instance
        videos_per_channel: Number of videos to sample per channel
        output_path: Output file path (default: tests/data/raw_videos.json)

    Returns:
        PrepareResult with counts and output path
    """
    output_path = output_path or TEST_DATA_DIR / "raw_videos.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channels = db.list_channels()
    all_videos: list[dict] = []

    for channel in channels:
        # Get videos with transcripts (sections exist)
        videos = db.list_videos(channel_id=channel.id, status="fetched")

        # Filter to videos that have sections
        videos_with_sections = []
        for video in videos:
            sections = db.get_sections(video.id)
            if sections:
                videos_with_sections.append((video, sections))

        if not videos_with_sections:
            continue

        # Sample up to N videos per channel
        sampled = random.sample(
            videos_with_sections,
            min(videos_per_channel, len(videos_with_sections)),
        )

        for video, sections in sampled:
            raw = RawVideoData(
                video_id=video.id,
                title=video.title,
                channel_id=channel.id,
                channel_name=channel.name,
                duration_seconds=video.duration_seconds,
                view_count=video.view_count,
                sections=[
                    {
                        "id": s.id,
                        "title": s.title,
                        "content": s.content[:2000],  # Truncate for LLM context
                        "start_time": s.start_time,
                    }
                    for s in sections[:10]  # Limit sections per video
                ],
            )
            all_videos.append({
                "video_id": raw.video_id,
                "title": raw.title,
                "channel_id": raw.channel_id,
                "channel_name": raw.channel_name,
                "duration_seconds": raw.duration_seconds,
                "view_count": raw.view_count,
                "sections": raw.sections,
            })

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump({"videos": all_videos}, f, indent=2)

    return PrepareResult(
        channels_sampled=len(channels),
        videos_sampled=len(all_videos),
        output_file=output_path,
    )


# LLM prompt for video analysis (full video - used for local LLMs)
VIDEO_ANALYSIS_PROMPT = """\
Analyze this YouTube video transcript and extract structured information.

Video: {title}
Channel: {channel_name}

Sections:
{sections_text}

Extract the following as JSON:
{{
  "entities": ["list of product names, model numbers, brands, people mentioned"],
  "topics": ["main themes/subjects discussed (3-5 items)"],
  "facts": [
    {{"section_id": "...", "fact": "specific data point or claim", "keywords": ["key", "terms"]}}
  ],
  "comparisons": [
    {{"items": ["item1", "item2"], "aspect": "what's being compared"}}
  ],
  "howtos": [
    {{"action": "verb phrase", "subject": "what/how", "section_id": "..."}}
  ]
}}

Rules:
- entities: Only specific names (e.g., "Model 3", "F-150", "Tim Cook"), not generic terms
- topics: High-level themes (e.g., "electric vehicles", "reliability concerns")
- facts: Specific claims with numbers or definitive statements
- comparisons: Only explicit A vs B comparisons mentioned in content
- howtos: Only if the video explains how to do something

Return valid JSON only."""


# LLM prompt for batch section analysis (used for OpenAI - multiple sections per call)
BATCH_SECTION_PROMPT = """\
Extract structured info from these video sections. Return JSON only.

Video: {title} | Channel: {channel_name}

{sections_text}

Return JSON:
{{"sections": [
  {{"id": "section_id", "entities": ["specific names only"], "topics": ["1-2 themes"], "facts": [{{"fact": "claim", "keywords": ["terms"]}}], "comparisons": [{{"items": ["a","b"], "aspect": "what"}}], "howtos": [{{"action": "verb", "subject": "what"}}]}}
]}}

Rules: entities=specific names (Model 3, F-150), not generic. facts=claims with numbers. Empty arrays if none found."""


def _parse_llm_json(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown wrappers.

    Args:
        content: Raw LLM response content

    Returns:
        Parsed JSON dict
    """
    # Handle JSON wrapped in markdown
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return json.loads(content)


def _is_openai_model(model: str) -> bool:
    """Check if model is an OpenAI model (vs local Ollama).

    Args:
        model: Model name

    Returns:
        True if this is an OpenAI model
    """
    openai_prefixes = ("gpt-", "o1-", "o3-", "text-", "chatgpt-")
    return model.startswith(openai_prefixes) or "/" not in model and ":" not in model


def _analyze_sections_batch_openai(
    video_title: str,
    channel_name: str,
    sections: list[dict],
    model: str,
    batch_size: int = 5,
) -> list[dict]:
    """Analyze multiple sections in batches using OpenAI API.

    Args:
        video_title: Title of the video
        channel_name: Channel name
        sections: List of section dicts with id, title, content
        model: OpenAI model to use
        batch_size: Number of sections per API call (default 5)

    Returns:
        List of analysis dicts, one per section
    """
    all_analyses = []

    # Process sections in batches
    for i in range(0, len(sections), batch_size):
        batch = sections[i : i + batch_size]

        # Format sections for prompt - compact format
        sections_text = "\n\n".join([
            f"[{s['id']}] {s.get('title', 'Untitled')}\n{s.get('content', '')[:1500]}"
            for s in batch
        ])

        prompt = BATCH_SECTION_PROMPT.format(
            title=video_title,
            channel_name=channel_name,
            sections_text=sections_text,
        )

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.1,
                max_tokens=400 * len(batch),  # ~400 tokens per section
            )
            result = _parse_llm_json(response.content)

            # Extract section analyses and add section_id to nested items
            for section_analysis in result.get("sections", []):
                section_id = section_analysis.get("id", "")
                # Add section_id to facts and howtos
                for fact in section_analysis.get("facts", []):
                    fact["section_id"] = section_id
                for howto in section_analysis.get("howtos", []):
                    howto["section_id"] = section_id
                all_analyses.append(section_analysis)

        except Exception as e:
            logger.warning(f"Failed to analyze batch starting at {i}: {e}")
            # Continue with next batch instead of failing entirely
            continue

    return all_analyses


def _merge_section_analyses(analyses: list[dict]) -> dict:
    """Merge multiple section analyses into a single video analysis.

    Args:
        analyses: List of section analysis dicts

    Returns:
        Merged analysis with deduplicated entities/topics
    """
    all_entities: set[str] = set()
    all_topics: set[str] = set()
    all_facts: list[dict] = []
    all_comparisons: list[dict] = []
    all_howtos: list[dict] = []

    for analysis in analyses:
        # Entities - deduplicate
        for e in analysis.get("entities", []):
            if isinstance(e, str) and e:
                all_entities.add(e)
            elif isinstance(e, dict) and e.get("name"):
                all_entities.add(e["name"])

        # Topics - deduplicate
        for t in analysis.get("topics", []):
            if isinstance(t, str) and t:
                all_topics.add(t)

        # Facts - keep all (they have section context)
        all_facts.extend(analysis.get("facts", []))

        # Comparisons - keep unique by items
        for comp in analysis.get("comparisons", []):
            items = tuple(sorted(comp.get("items", [])))
            if items and not any(
                tuple(sorted(c.get("items", []))) == items for c in all_comparisons
            ):
                all_comparisons.append(comp)

        # Howtos - keep all (they have section context)
        all_howtos.extend(analysis.get("howtos", []))

    return {
        "entities": list(all_entities),
        "topics": list(all_topics)[:5],  # Limit to 5 main topics
        "facts": all_facts,
        "comparisons": all_comparisons,
        "howtos": all_howtos,
    }


def analyze_videos(
    input_path: Path | None = None,
    output_path: Path | None = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    limit: int | None = None,
    use_openai: bool = False,
) -> AnalyzeResult:
    """Analyze videos using LLM to extract entities, topics, facts.

    For local models (Ollama): Analyzes multiple sections at once.
    For OpenAI models: Analyzes each section individually to reduce cost,
    then merges results.

    Args:
        input_path: Path to raw_videos.json (default: tests/data/raw_videos.json)
        output_path: Output file path (default: tests/data/video_analysis_{model}.json)
        model: LLM model to use for analysis
        limit: Max videos to analyze (None = all)
        use_openai: Force OpenAI API even if model name doesn't match

    Returns:
        AnalyzeResult with counts and output path
    """
    input_path = input_path or TEST_DATA_DIR / "raw_videos.json"

    # Auto-detect output path based on model name if not specified
    if output_path is None:
        output_path = TEST_DATA_DIR / get_analysis_filename(model)

    with open(input_path) as f:
        raw_data = json.load(f)

    videos = raw_data["videos"]
    if limit:
        videos = videos[:limit]

    # Determine if we should use OpenAI API
    is_openai = use_openai or _is_openai_model(model)

    analyses: list[dict] = []
    total_entities = 0
    total_topics = 0
    total_facts = 0

    for video in videos:
        video_id = video["video_id"]
        title = video["title"]
        channel_name = video["channel_name"]
        sections = video["sections"]

        if is_openai:
            # OpenAI: Batch analyze sections (5 per API call) to reduce cost
            section_analyses = _analyze_sections_batch_openai(
                video_title=title,
                channel_name=channel_name,
                sections=sections[:10],  # Limit to 10 sections
                model=model,
                batch_size=5,
            )

            if not section_analyses:
                logger.warning(f"No sections analyzed for {video_id}")
                continue

            # Merge all section analyses
            merged = _merge_section_analyses(section_analyses)
            result = {
                "video_id": video_id,
                "title": title,
                "channel_name": channel_name,
                **merged,
            }

        else:
            # Ollama: Analyze multiple sections at once (cheaper, faster locally)
            sections_text = "\n\n".join([
                f"[{s['id']}] {s.get('title', '')}\n{s.get('content', '')[:1000]}"
                for s in sections[:5]  # Limit sections in prompt
            ])

            prompt = VIDEO_ANALYSIS_PROMPT.format(
                title=title,
                channel_name=channel_name,
                sections_text=sections_text,
            )

            try:
                response = ollama_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.1,
                )

                analysis = _parse_llm_json(response.content)

                result = {
                    "video_id": video_id,
                    "title": title,
                    "channel_name": channel_name,
                    "entities": analysis.get("entities", []),
                    "topics": analysis.get("topics", []),
                    "facts": analysis.get("facts", []),
                    "comparisons": analysis.get("comparisons", []),
                    "howtos": analysis.get("howtos", []),
                }

            except Exception as e:
                logger.warning(f"Failed to analyze {video_id}: {e}")
                continue

        analyses.append(result)

        total_entities += len(result.get("entities", []))
        total_topics += len(result.get("topics", []))
        total_facts += len(result.get("facts", []))

        logger.info(
            f"Analyzed: {title[:50]}... - "
            f"{len(result.get('entities', []))} entities, "
            f"{len(result.get('topics', []))} topics, "
            f"{len(result.get('facts', []))} facts"
        )

    # Save to JSON with metadata
    output_data = {
        "metadata": {
            "model": model,
            "is_openai": is_openai,
            "videos_analyzed": len(analyses),
        },
        "analyses": analyses,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return AnalyzeResult(
        videos_analyzed=len(analyses),
        total_entities=total_entities,
        total_topics=total_topics,
        total_facts=total_facts,
        output_file=output_path,
    )


def build_tests(
    input_path: Path | None = None,
    output_path: Path | None = None,
    tests_per_type_per_channel: int = 5,
    model: str | None = None,
) -> BuildResult:
    """Generate structured test cases from video analysis.

    Creates test queries organized by channel with balanced coverage:
    - 5 entity queries per channel (product names, brands, people)
    - 5 topic queries per channel (themes, subjects)
    - 5 comparison queries per channel (X vs Y)
    - 5 list/count queries per channel (aggregation queries)

    Each test includes verification metadata (video_id, section_id) for recall.

    Args:
        input_path: Path to video_analysis.json (or video_analysis_{model}.json)
        output_path: Output benchmark file path (auto-generated if model specified)
        tests_per_type_per_channel: Number of tests per type per channel (default 5)
        model: Model name used for analysis (for auto-naming output file)

    Returns:
        BuildResult with counts and output path
    """
    # Handle input path - if model specified and no input_path, use model-named file
    if input_path is None:
        if model:
            input_path = TEST_DATA_DIR / get_analysis_filename(model)
        else:
            input_path = TEST_DATA_DIR / "video_analysis.json"

    # Handle output path - if model specified and no output_path, use model-named file
    if output_path is None:
        if model:
            output_path = TEST_DATA_DIR / get_benchmark_filename(model)
        else:
            output_path = TEST_DATA_DIR / "benchmark_generated.json"

    with open(input_path) as f:
        analysis_data = json.load(f)

    metadata = analysis_data.get("metadata", {})
    source_model = metadata.get("model", model or "unknown")

    # Group analyses by channel
    by_channel: dict[str, list[dict]] = {}
    for analysis in analysis_data["analyses"]:
        channel = analysis.get("channel_name", "unknown")
        if channel not in by_channel:
            by_channel[channel] = []
        by_channel[channel].append(analysis)

    test_cases: list[dict] = []
    by_type: dict[str, int] = {}
    n = tests_per_type_per_channel

    for channel, analyses in by_channel.items():
        # Collect all data from this channel's videos
        channel_entities: list[dict] = []  # [{entity, video_id, title}]
        channel_topics: list[dict] = []
        channel_comparisons: list[dict] = []
        channel_facts: list[dict] = []

        for analysis in analyses:
            video_id = analysis["video_id"]
            title = analysis["title"]

            # Entities
            for e in analysis.get("entities", []):
                if isinstance(e, dict):
                    e = e.get("name", "")
                if isinstance(e, str) and e and len(e) > 2:
                    channel_entities.append({
                        "entity": e,
                        "video_id": video_id,
                        "title": title,
                        "channel": channel,
                    })

            # Topics
            for t in analysis.get("topics", []):
                if isinstance(t, str) and t:
                    channel_topics.append({
                        "topic": t,
                        "video_id": video_id,
                        "title": title,
                        "channel": channel,
                    })

            # Comparisons
            for comp in analysis.get("comparisons", []):
                items = comp.get("items", [])
                if len(items) >= 2:
                    channel_comparisons.append({
                        "items": items,
                        "aspect": comp.get("aspect", ""),
                        "video_id": video_id,
                        "title": title,
                        "channel": channel,
                    })

            # Facts (for list/count queries)
            for fact in analysis.get("facts", []):
                if fact.get("keywords"):
                    channel_facts.append({
                        "fact": fact.get("fact", ""),
                        "keywords": fact.get("keywords", []),
                        "section_id": fact.get("section_id"),
                        "video_id": video_id,
                        "title": title,
                        "channel": channel,
                    })

        # Generate entity tests (sample up to n unique entities)
        seen_entities = set()
        entity_samples = []
        random.shuffle(channel_entities)
        for item in channel_entities:
            e_lower = item["entity"].lower()
            if e_lower not in seen_entities:
                seen_entities.add(e_lower)
                entity_samples.append(item)
            if len(entity_samples) >= n:
                break

        for item in entity_samples:
            test_cases.append({
                "query": item["entity"],
                "expected_type": "entity",
                "expected_keywords": [item["entity"].lower()],
                "expected_video_ids": [item["video_id"]],
                "channel": item["channel"],
                "source_video": item["title"],
                "note": f"Entity from {channel}: {item['title'][:30]}",
            })
            by_type["entity"] = by_type.get("entity", 0) + 1

        # Generate topic tests (sample up to n unique topics)
        seen_topics = set()
        topic_samples = []
        random.shuffle(channel_topics)
        for item in channel_topics:
            t_lower = item["topic"].lower()
            if t_lower not in seen_topics:
                seen_topics.add(t_lower)
                topic_samples.append(item)
            if len(topic_samples) >= n:
                break

        for item in topic_samples:
            test_cases.append({
                "query": item["topic"],
                "expected_type": "topic",
                "expected_keywords": item["topic"].lower().split()[:3],
                "expected_video_ids": [item["video_id"]],
                "channel": item["channel"],
                "source_video": item["title"],
                "note": f"Topic from {channel}: {item['title'][:30]}",
            })
            by_type["topic"] = by_type.get("topic", 0) + 1

        # Generate comparison tests (sample up to n)
        random.shuffle(channel_comparisons)
        for item in channel_comparisons[:n]:
            items = item["items"]
            query = f"{items[0]} vs {items[1]}"
            test_cases.append({
                "query": query,
                "expected_type": "comparison",
                "expected_keywords": [i.lower() for i in items[:2]],
                "expected_video_ids": [item["video_id"]],
                "channel": item["channel"],
                "source_video": item["title"],
                "note": f"Comparison from {channel}: {item['aspect']}",
            })
            by_type["comparison"] = by_type.get("comparison", 0) + 1

        # Generate list/count tests based on entities
        # "How many videos about X?" where X is an entity in this channel
        count_entities = random.sample(
            list(seen_entities), min(n, len(seen_entities))
        )
        for entity in count_entities:
            # Find all videos with this entity
            matching_videos = [
                e["video_id"] for e in channel_entities
                if e["entity"].lower() == entity
            ]
            test_cases.append({
                "query": f"how many videos about {entity}?",
                "expected_type": "list",
                "expected_keywords": [entity],
                "expected_video_ids": list(set(matching_videos)),
                "expected_count": len(set(matching_videos)),
                "channel": channel,
                "note": f"Count query for {channel}",
            })
            by_type["list"] = by_type.get("list", 0) + 1

    # Add META queries (library stats) - these are global, not per-channel
    meta_queries = [
        {
            "query": "how many videos are there?",
            "expected_type": "meta",
            "expected_keywords": [],
            "expected_video_ids": [],
            "channel": None,
            "note": "Library video count",
        },
        {
            "query": "what channels?",
            "expected_type": "meta",
            "expected_keywords": [],
            "expected_video_ids": [],
            "channel": None,
            "note": "List channels",
        },
        {
            "query": "how many channels?",
            "expected_type": "meta",
            "expected_keywords": [],
            "expected_video_ids": [],
            "channel": None,
            "note": "Channel count",
        },
    ]
    for mq in meta_queries:
        test_cases.append(mq)
        by_type["meta"] = by_type.get("meta", 0) + 1

    # Save to JSON
    output = {
        "description": "Structured benchmark with balanced coverage per channel",
        "version": "3.0",
        "analysis_model": source_model,
        "generated_from": str(input_path),
        "tests_per_type_per_channel": n,
        "channels": list(by_channel.keys()),
        "test_cases": test_cases,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return BuildResult(
        tests_generated=len(test_cases),
        by_type=by_type,
        output_file=output_path,
    )
