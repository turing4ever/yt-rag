"""CLI commands for yt-rag."""

import asyncio
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import __version__
from .chapters import sectionize_video
from .config import (
    DB_PATH,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FETCH_WORKERS,
    DEFAULT_OLLAMA_MODEL,
    METADATA_FRESHNESS_DAYS,
    RESTRICTED_AVAILABILITY,
    ensure_data_dir,
    get_yt_dlp_batch_size,
    get_yt_dlp_delay,
)
from .db import Database
from .discovery import (
    extract_video_id,
    get_channel_info,
    get_video_info,
    list_channel_videos,
)
from .embed import embed_all_sections, embed_all_summaries, embed_video, get_index_stats
from .eval import add_feedback, add_test_case, run_benchmark
from .export import export_all_chunks, export_to_json, export_to_jsonl
from .models import Channel, Video
from .search import search as rag_search
from .summarize import summarize_video
from .transcript import TranscriptUnavailable, fetch_transcript

# Configure logging based on LOG_LEVEL environment variable
_log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(levelname)s %(name)s: %(message)s",
)
# Suppress noisy FAISS loader debug messages
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logging.getLogger("faiss._loader").setLevel(logging.WARNING)

app = typer.Typer(
    name="yt-rag",
    help="Extract YouTube transcripts for RAG pipelines.",
    no_args_is_help=True,
)
console = Console()


@dataclass
class MetadataRefreshResult:
    """Result of metadata refresh operation."""

    channels_updated: int = 0
    channels_errors: int = 0
    videos_updated: int = 0
    videos_errors: int = 0


async def refresh_metadata_async(
    work_items: list[tuple[str, Channel | Video]],
    db: Database,
    batch_size: int | None = None,
) -> MetadataRefreshResult:
    """Refresh metadata for channels and videos with async batching.

    Args:
        work_items: List of (type, item) tuples where type is "channel" or "video"
        db: Database instance for saving results
        batch_size: Items per batch (default from config)

    Returns:
        MetadataRefreshResult with counts
    """
    if not work_items:
        return MetadataRefreshResult()

    batch_size = batch_size or get_yt_dlp_batch_size()
    result = MetadataRefreshResult()

    async def fetch_one(item_type: str, item: Channel | Video, index: int):
        # Stagger requests within batch to avoid simultaneous hits
        await asyncio.sleep(get_yt_dlp_delay() * index)
        try:
            if item_type == "channel":
                data = await asyncio.to_thread(get_channel_info, item.url)
            else:
                data = await asyncio.to_thread(get_video_info, item.url)
            return (item_type, item, "success", data)
        except Exception as e:
            return (item_type, item, "error", str(e))

    async def process_batch(batch: list, progress, task):
        tasks = [fetch_one(t, item, i) for i, (t, item) in enumerate(batch)]
        results = await asyncio.gather(*tasks)

        for item_type, item, status, data in results:
            if item_type == "channel":
                if status == "success":
                    db.add_channel(data)
                    result.channels_updated += 1
                    progress.update(task, description=f"[green]✓[/green] {item.name}")
                else:
                    result.channels_errors += 1
                    console.print(f"\n[red]Error[/red] {item.name}: {data}")
            else:
                title = item.title[:40] + "..." if len(item.title) > 40 else item.title
                if status == "success":
                    db.add_video(data)
                    result.videos_updated += 1
                    progress.update(task, description=f"[green]✓[/green] {title}")
                else:
                    result.videos_errors += 1
                    console.print(f"\n[red]Error[/red] {title}: {data}")
            progress.advance(task)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Refreshing metadata", total=len(work_items))
        for i in range(0, len(work_items), batch_size):
            batch = work_items[i : i + batch_size]
            await process_batch(batch, progress, task)

    return result


def select_test_videos(
    db: Database, channels: list[Channel], per_channel: int = 5
) -> tuple[set[str], dict[str, tuple[int, int]]]:
    """Select test videos from each channel.

    Prefers pending videos, fills remainder with fetched.

    Returns:
        (set of video IDs, dict of channel_name -> (new_count, existing_count))
    """
    test_ids: set[str] = set()
    stats: dict[str, tuple[int, int]] = {}

    pending_all = db.get_pending_videos()
    fetched_all = db.list_videos(status="fetched")

    for channel in channels:
        pending = [v for v in pending_all if v.channel_id == channel.id]
        fetched = [v for v in fetched_all if v.channel_id == channel.id]

        selected = pending[:per_channel]
        remaining = per_channel - len(selected)
        if remaining > 0:
            selected.extend(fetched[:remaining])

        test_ids.update(v.id for v in selected)
        new_count = min(len(pending), per_channel)
        stats[channel.name] = (new_count, len(selected) - new_count)

    return test_ids, stats


def get_db() -> Database:
    """Get database instance."""
    db = Database()
    db.init()
    return db


@app.command()
def init():
    """Initialize the database."""
    ensure_data_dir()
    db = get_db()
    db.close()
    console.print(f"[green]✓[/green] Created database at {DB_PATH}")


@app.command()
def update(
    force_transcript: bool = typer.Option(
        False, "--force-transcript", help="Re-fetch all transcripts, not just pending"
    ),
    force_meta: bool = typer.Option(
        False, "--force-meta", help="Force refresh all metadata, ignoring timestamps"
    ),
    force_embed: bool = typer.Option(
        False, "--force-embed", help="Rebuild all embeddings from scratch"
    ),
    force_synonym: bool = typer.Option(
        False, "--force-synonym", help="Regenerate synonyms for all videos"
    ),
    skip_sync: bool = typer.Option(False, "--skip-sync", help="Skip syncing channels"),
    skip_meta: bool = typer.Option(False, "--skip-meta", help="Skip metadata refresh"),
    skip_embed: bool = typer.Option(False, "--skip-embed", help="Skip embedding step"),
    skip_synonym: bool = typer.Option(False, "--skip-synonym", help="Skip synonym generation"),
    test: bool = typer.Option(
        False, "--test", help="Test mode: process 5 videos per channel through entire pipeline"
    ),
    workers: int = typer.Option(
        DEFAULT_FETCH_WORKERS, "-w", "--workers", help="Parallel workers for transcript fetch"
    ),
    model: str = typer.Option(None, "-m", "--model", help="Override default LLM model"),
    openai: bool = typer.Option(False, "--openai", help="Use OpenAI API instead of local Ollama"),
):
    """Run the full update pipeline.

    This command runs the complete pipeline to update your library:
    1. sync-channel: Pull new videos from tracked channels
    2. refresh-meta: Refresh metadata (skips if refreshed within 1 day)
    3. fetch-transcript: Fetch transcripts for pending videos
    4. process-transcript: Sectionize and summarize videos
    5. embed: Build/update vector index
    6. synonyms: Generate synonyms for new videos (or all with --force-synonym)

    By default uses local Ollama. Use --openai to use OpenAI API.
    Use --model to override the default model for either backend.

    Examples:
        yt-rag update                      # Run full pipeline (local Ollama)
        yt-rag update --openai             # Use OpenAI API for all steps
        yt-rag update --openai --model gpt-4o  # Use specific OpenAI model
        yt-rag update --model qwen3:8b     # Use specific local model
        yt-rag update --test               # Test run: 5 videos per channel
        yt-rag update --skip-sync          # Skip channel sync
        yt-rag update --force-embed        # Rebuild all embeddings
    """
    from datetime import datetime, timedelta

    from .openai_client import check_ollama_running

    use_local = not openai

    if test:
        console.print("[yellow]Test mode: 5 videos per channel[/yellow]")

    if use_local and not check_ollama_running():
        console.print("[red]Error: Ollama is not running[/red]")
        console.print("Start it with: sudo systemctl start ollama")
        console.print("Or use --openai to use OpenAI embeddings")
        raise typer.Exit(1)

    db = get_db()
    channels = db.list_channels()

    # Select test videos upfront
    test_video_ids: set[str] = set()
    if test:
        console.print("\n[bold]Selecting test videos[/bold]")
        test_video_ids, stats = select_test_videos(db, channels)
        for name, (new, existing) in stats.items():
            console.print(f"  {name}: {new} new + {existing} existing = {new + existing}")
        console.print(f"[dim]Total: {len(test_video_ids)}[/dim]")

    # Step 1: Sync channels
    if not skip_sync:
        console.print("\n[bold]Step 1: Syncing channels[/bold]")
        if not channels:
            console.print("[yellow]No channels tracked. Use 'yt-rag add <url>' first.[/yellow]")
        else:
            total_new = 0
            for channel in channels:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task(f"Syncing {channel.name}...", total=None)
                    videos = list_channel_videos(channel.url, channel.id)
                    added = db.add_videos(videos)
                    db.update_channel_sync_time(channel.id)
                    total_new += added
                console.print(f"  {channel.name}: {added} new videos")
            console.print(
                f"[green]✓[/green] Synced {len(channels)} channels, {total_new} new videos"
            )
    else:
        console.print("\n[bold]Step 1: Syncing channels[/bold] [dim](skipped)[/dim]")

    # Step 2: Refresh metadata
    if not skip_meta:
        console.print("\n[bold]Step 2: Refreshing metadata[/bold]")
        cutoff = datetime.now() - timedelta(days=METADATA_FRESHNESS_DAYS)

        channels_to_refresh = [c for c in db.list_channels() if not c.description]
        all_videos = db.list_videos(status="fetched")
        pending_videos = db.get_pending_videos()

        if test:
            videos_to_refresh = [v for v in all_videos + pending_videos if v.id in test_video_ids]
            console.print(f"[dim]Test mode: {len(videos_to_refresh)} videos[/dim]")
        elif force_meta:
            videos_to_refresh = all_videos
            console.print(f"[yellow]Force: {len(videos_to_refresh)} videos[/yellow]")
        else:
            videos_to_refresh = [
                v
                for v in all_videos
                if v.metadata_refreshed_at is None or v.metadata_refreshed_at < cutoff
            ]
            skipped = len(all_videos) - len(videos_to_refresh)
            if skipped > 0:
                console.print(f"[dim]Skipping {skipped} fresh videos[/dim]")

        work_items: list[tuple[str, Channel | Video]] = []
        work_items.extend([("channel", c) for c in channels_to_refresh])
        work_items.extend([("video", v) for v in videos_to_refresh])

        if not work_items:
            console.print("[green]✓[/green] All metadata up to date")
        else:
            console.print(
                f"[dim]{len(channels_to_refresh)} channels + {len(videos_to_refresh)} videos[/dim]"
            )
            result = asyncio.run(refresh_metadata_async(work_items, db))
            parts = []
            if result.channels_updated or result.channels_errors:
                parts.append(f"{result.channels_updated} channels")
            if result.videos_updated or result.videos_errors:
                parts.append(f"{result.videos_updated} videos")
            total_errors = result.channels_errors + result.videos_errors
            error_msg = f", {total_errors} errors" if total_errors else ""
            console.print(f"[green]✓[/green] Updated {', '.join(parts)}{error_msg}")
    else:
        console.print("\n[bold]Step 2: Refreshing metadata[/bold] [dim](skipped)[/dim]")

    # Step 3: Fetch transcripts (uses youtube_transcript_api - parallel, no rate limiting)
    console.print("\n[bold]Step 3: Fetching transcripts[/bold]")

    def is_fetchable(v: Video) -> bool:
        """Check if video transcript is likely accessible."""
        return v.availability not in RESTRICTED_AVAILABILITY if v.availability else True

    if test:
        videos_to_fetch = [v for v in db.get_pending_videos() if v.id in test_video_ids]
        console.print(f"[dim]Test mode: {len(videos_to_fetch)} pending[/dim]")
    elif force_transcript:
        videos_to_fetch = db.list_videos(status="fetched")
        videos_to_fetch.extend(db.get_pending_videos())
        console.print(f"[yellow]Force: {len(videos_to_fetch)} videos[/yellow]")
    else:
        videos_to_fetch = db.get_pending_videos()

    # Filter out restricted videos
    restricted = [v for v in videos_to_fetch if not is_fetchable(v)]
    videos_to_fetch = [v for v in videos_to_fetch if is_fetchable(v)]
    if restricted:
        console.print(f"[dim]Skipping {len(restricted)} restricted videos[/dim]")
        # Mark them as unavailable
        for v in restricted:
            db.update_video_status(v.id, "unavailable")

    if not videos_to_fetch:
        console.print("[green]✓[/green] No pending videos to fetch")
    else:
        fetched = 0
        unavailable = 0
        errors = 0
        lock = threading.Lock()

        def process_video_fetch(video):
            """Fetch transcript only - metadata is handled separately in Step 2."""
            try:
                transcript = fetch_transcript(video.id)
                return (video, "fetched", transcript)
            except TranscriptUnavailable:
                return (video, "unavailable", None)
            except Exception as e:
                return (video, "error", str(e))

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching", total=len(videos_to_fetch))

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_video_fetch, v): v for v in videos_to_fetch}

                for future in as_completed(futures):
                    video, status, result = future.result()
                    title = video.title[:40] + "..." if len(video.title) > 40 else video.title

                    with lock:
                        if status == "fetched":
                            db.add_segments(result.segments)
                            db.update_video_status(video.id, "fetched")
                            fetched += 1
                            progress.update(task, description=f"[green]✓[/green] {title}")
                        elif status == "unavailable":
                            db.update_video_status(video.id, "unavailable")
                            unavailable += 1
                        elif status == "error":
                            db.update_video_status(video.id, "error")
                            errors += 1

                        progress.advance(task)

        console.print(
            f"[green]✓[/green] Fetched {fetched}, unavailable {unavailable}, errors {errors}"
        )

    # Step 4: Process transcripts (sectionize + summarize)
    console.print("\n[bold]Step 4: Processing transcripts[/bold]")
    videos = db.list_videos(status="fetched")

    # Filter to videos needing processing
    videos_to_process = []
    for v in videos:
        # In test mode, only process test videos
        if test and v.id not in test_video_ids:
            continue
        existing_sections = db.get_sections(v.id)
        existing_summary = db.get_summary(v.id)
        # In test mode, force reprocess even if already done
        if test or (not existing_sections or not existing_summary):
            videos_to_process.append(v)

    if test:
        console.print(f"[dim]Test mode: {len(videos_to_process)} videos to process[/dim]")

    if not videos_to_process:
        console.print("[green]✓[/green] All videos already processed")
    else:
        sectionized = 0
        summarized = 0
        errors = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing", total=len(videos_to_process))

            for video in videos_to_process:
                title = video.title[:35] + "..." if len(video.title) > 35 else video.title

                # Sectionize if needed
                existing_sections = db.get_sections(video.id)
                if not existing_sections:
                    try:
                        progress.update(task, description=f"Sectionizing: {title}")
                        sectionize_video(video.id, db, model=model, use_openai=openai)
                        sectionized += 1
                    except Exception as e:
                        console.print(f"\n[red]Sectionize error[/red] {video.id}: {e}")
                        errors += 1
                        progress.advance(task)
                        continue

                # Summarize if needed
                existing_summary = db.get_summary(video.id)
                if not existing_summary:
                    try:
                        progress.update(task, description=f"Summarizing: {title}")
                        summarize_video(video.id, db, model=model, use_openai=openai)
                        summarized += 1
                    except Exception as e:
                        console.print(f"\n[red]Summarize error[/red] {video.id}: {e}")
                        errors += 1

                progress.advance(task)

        console.print(
            f"[green]✓[/green] Sectionized {sectionized}, summarized {summarized}, errors {errors}"
        )

    # Step 5: Embed
    if not skip_embed:
        console.print("\n[bold]Step 5: Building embeddings[/bold]")
        backend_name = "local (Ollama)" if use_local else "OpenAI"
        console.print(f"[dim]Using {backend_name} embeddings[/dim]")

        if test:
            # In test mode, embed only test videos
            console.print(f"[dim]Test mode: embedding {len(test_video_ids)} videos[/dim]")
            total_sections = 0
            total_summaries = 0
            total_tokens = 0

            for video_id in test_video_ids:
                sections = db.get_sections(video_id)
                if sections:
                    result = embed_video(video_id, db, model=None, force=True, use_local=use_local)
                    total_sections += result.items_embedded
                    total_tokens += result.tokens_used
                summary = db.get_summary(video_id)
                if summary:
                    total_summaries += 1

            console.print(
                f"[green]✓[/green] Embedded {total_sections} sections, "
                f"{total_summaries} summaries ({total_tokens} tokens)"
            )
        else:
            if force_embed:
                console.print("[yellow]Force mode: rebuilding all embeddings[/yellow]")

            stats = db.get_stats()
            if stats["sections"] == 0:
                console.print("[yellow]No sections to embed[/yellow]")
            else:
                # Embed sections with progress bar
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Embedding sections...", total=None)

                    def update_progress(embedded: int, total: int) -> None:
                        if progress.tasks[task].total is None:
                            progress.update(task, total=total)
                        progress.update(task, completed=embedded)

                    result = embed_all_sections(
                        db,
                        model=None,
                        rebuild=force_embed,
                        use_local=use_local,
                        progress_callback=update_progress,
                    )

                if result.items_embedded > 0:
                    console.print(
                        f"[green]✓[/green] Embedded {result.items_embedded} sections "
                        f"({result.tokens_used} tokens)"
                    )
                else:
                    console.print("[green]✓[/green] All sections already embedded")

                # Embed summaries
                if stats["summaries"] > 0:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        progress.add_task("Embedding summaries...", total=None)
                        summary_result = embed_all_summaries(
                            db, model=None, rebuild=force_embed, use_local=use_local
                        )

                    if summary_result.items_embedded > 0:
                        embed_count = summary_result.items_embedded
                        tokens = summary_result.tokens_used
                        console.print(
                            f"[green]✓[/green] Embedded {embed_count} summaries ({tokens} tokens)"
                        )
                    else:
                        console.print("[green]✓[/green] All summaries already embedded")
    else:
        console.print("\n[bold]Step 5: Building embeddings[/bold] [dim](skipped)[/dim]")

    # Step 6: Refresh synonyms
    if not skip_synonym:
        console.print("\n[bold]Step 6: Generating synonyms[/bold]")
        from .keywords import refresh_synonyms

        if test:
            console.print(f"[dim]Test mode: processing {len(test_video_ids)} videos[/dim]")
            # In test mode, extract keywords from test videos only
            from .keywords import extract_keywords_from_videos, suggest_synonyms_for_keyword

            keywords = extract_keywords_from_videos(db, list(test_video_ids), min_total_frequency=3)
            if keywords:
                for kw in keywords[:50]:
                    synonyms = suggest_synonyms_for_keyword(kw.keyword)
                    for syn in synonyms:
                        db.add_synonym(kw.keyword, syn, source="heuristic", approved=False)
                msg = f"[green]✓[/green] Extracted {len(keywords)} keywords from test videos"
                console.print(msg)
            else:
                console.print("[green]✓[/green] No keywords extracted")
        else:
            if force_synonym:
                console.print("[yellow]Force mode: regenerating all synonyms[/yellow]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Extracting keywords and generating synonyms...", total=None)
                syn_result = refresh_synonyms(db, force=force_synonym, use_local=use_local)

            if syn_result.videos_analyzed > 0:
                console.print(
                    f"[green]✓[/green] Analyzed {syn_result.videos_analyzed} videos, "
                    f"extracted {syn_result.keywords_extracted} keywords, "
                    f"added {syn_result.synonyms_added} synonyms"
                )
                if syn_result.channels_processed:
                    channels = ", ".join(syn_result.channels_processed)
                    console.print(f"[dim]Channels: {channels}[/dim]")
            else:
                console.print("[green]✓[/green] No new videos to analyze")
    else:
        console.print("\n[bold]Step 6: Generating synonyms[/bold] [dim](skipped)[/dim]")

    db.close()
    console.print("\n[bold green]✓ Update complete![/bold green]")


@app.command()
def add(url: str):
    """Add a YouTube channel or video to track."""
    db = get_db()

    # Check if it's a video URL
    video_id = extract_video_id(url)
    if video_id and "/watch?" in url:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Fetching video info...", total=None)
            video = get_video_info(url)
            db.add_video(video)

        console.print(f"[green]✓[/green] Added video: {video.title}")
        db.close()
        return

    # It's a channel URL
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching channel info...", total=None)
        channel = get_channel_info(url)
        db.add_channel(channel)

        progress.update(task, description=f"Listing videos from {channel.name}...")
        videos = list_channel_videos(url, channel.id)
        added = db.add_videos(videos)

    msg = f"[green]✓[/green] Added channel: {channel.name} ({len(videos)} videos, {added} new)"
    console.print(msg)
    db.close()


@app.command("sync-channel")
def sync_channel():
    """Sync videos from all tracked channels."""
    db = get_db()
    channels = db.list_channels()

    if not channels:
        console.print("[yellow]No channels tracked. Use 'yt-rag add <url>' first.[/yellow]")
        db.close()
        return

    total_new = 0
    for channel in channels:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Syncing {channel.name}...", total=None)
            videos = list_channel_videos(channel.url, channel.id)
            added = db.add_videos(videos)
            db.update_channel_sync_time(channel.id)
            total_new += added

        console.print(f"  {channel.name}: {added} new videos")

    console.print(f"[green]✓[/green] Synced {len(channels)} channels, {total_new} new videos")
    db.close()


@app.command("fetch-transcript")
def fetch_transcript_cmd(
    limit: int = typer.Option(None, help="Max videos to fetch"),
    workers: int = typer.Option(
        DEFAULT_FETCH_WORKERS, "-w", "--workers", help="Number of parallel workers"
    ),
):
    """Fetch transcripts for pending videos."""
    db = get_db()
    pending = db.get_pending_videos()

    if not pending:
        console.print("[green]✓[/green] No pending videos")
        db.close()
        return

    if limit:
        pending = pending[:limit]

    fetched = 0
    unavailable = 0
    errors = 0
    lock = threading.Lock()
    interrupted = False

    def process_video(video):
        """Fetch transcript for a single video. Returns (video, result, error, updated_video)."""
        nonlocal interrupted
        if interrupted:
            return (video, "skipped", None, None)
        try:
            # Fetch full metadata if missing
            updated_video = None
            if not video.description:
                try:
                    updated_video = get_video_info(video.url)
                except Exception:
                    pass  # Keep going without metadata update

            transcript = fetch_transcript(video.id)
            return (video, "fetched", transcript, updated_video)
        except TranscriptUnavailable:
            return (video, "unavailable", None, None)
        except Exception as e:
            return (video, "error", str(e), None)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching", total=len(pending))

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_video, v): v for v in pending}

                for future in as_completed(futures):
                    if interrupted:
                        break

                    video, status, result, updated_video = future.result()
                    title = video.title[:40] + "..." if len(video.title) > 40 else video.title

                    with lock:
                        if status == "fetched":
                            # Update video metadata if we fetched it
                            if updated_video:
                                db.add_video(updated_video)
                            db.add_segments(result.segments)
                            db.update_video_status(video.id, "fetched")
                            fetched += 1
                            progress.update(task, description=f"[green]✓[/green] {title}")
                        elif status == "unavailable":
                            db.update_video_status(video.id, "unavailable")
                            unavailable += 1
                            progress.update(task, description=f"[yellow]○[/yellow] {title}")
                        elif status == "error":
                            db.update_video_status(video.id, "error")
                            errors += 1
                            console.print(f"\n[red]Error[/red] {video.title}: {result}")

                        progress.advance(task)
        except KeyboardInterrupt:
            interrupted = True
            console.print("\n[yellow]Interrupted, waiting for active tasks...[/yellow]")

    console.print(f"[green]✓[/green] Fetched {fetched}, unavailable {unavailable}, errors {errors}")
    db.close()


@app.command()
def export(
    output: Path = typer.Option(..., "-o", "--output", help="Output file path"),
    format: str = typer.Option("jsonl", "-f", "--format", help="Output format (jsonl, json)"),
    chunk_size: int = typer.Option(DEFAULT_CHUNK_SIZE, help="Words per chunk"),
    overlap: int = typer.Option(DEFAULT_CHUNK_OVERLAP, help="Overlap between chunks"),
    channel: str = typer.Option(None, help="Filter by channel ID"),
):
    """Export transcripts for RAG pipeline."""
    db = get_db()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Exporting chunks...", total=None)
        chunks = export_all_chunks(db, chunk_size, overlap, channel)

        if format == "jsonl":
            count = export_to_jsonl(chunks, output)
        else:
            count = export_to_json(chunks, output)

    console.print(f"[green]✓[/green] Exported {count} chunks to {output}")
    db.close()


@app.command("list")
def list_items(
    item_type: str = typer.Argument("channels", help="What to list (channels, videos)"),
    channel: str = typer.Option(None, help="Filter videos by channel ID"),
    status: str = typer.Option(None, help="Filter videos by status"),
):
    """List channels or videos."""
    db = get_db()

    if item_type == "channels":
        channels = db.list_channels()
        if not channels:
            console.print("No channels tracked")
            db.close()
            return

        table = Table(title="Channels")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Last Synced")

        for ch in channels:
            synced = ch.last_synced_at.strftime("%Y-%m-%d %H:%M") if ch.last_synced_at else "Never"
            table.add_row(ch.id, ch.name, synced)

        console.print(table)

    elif item_type == "videos":
        videos = db.list_videos(channel_id=channel, status=status)
        if not videos:
            console.print("No videos found")
            db.close()
            return

        table = Table(title=f"Videos ({len(videos)})")
        table.add_column("ID")
        table.add_column("Title", max_width=50)
        table.add_column("Status")

        for v in videos:
            status_color = {"fetched": "green", "pending": "yellow", "unavailable": "red"}.get(
                v.transcript_status, "white"
            )
            table.add_row(v.id, v.title[:50], f"[{status_color}]{v.transcript_status}[/]")

        console.print(table)

    else:
        console.print(f"[red]Unknown item type: {item_type}[/red]")

    db.close()


@app.command()
def status():
    """Show database statistics."""
    from .config import DEFAULT_OLLAMA_EMBED_MODEL
    from .vectorstore import VectorStore

    db = get_db()
    stats = db.get_stats()

    table = Table(title="yt-rag Status")
    table.add_column("Metric")
    table.add_column("Count", justify="right")

    table.add_row("Channels", str(stats["channels"]))
    table.add_row("Videos (total)", str(stats["videos_total"]))
    table.add_row("  Fetched", f"[green]{stats['videos_fetched']}[/green]")
    table.add_row("  Pending", f"[yellow]{stats['videos_pending']}[/yellow]")
    table.add_row("  Unavailable", f"[red]{stats['videos_unavailable']}[/red]")
    table.add_row("Segments", str(stats["segments"]))
    table.add_row("Sections", str(stats["sections"]))
    table.add_row("Summaries", str(stats["summaries"]))

    # Add embedding index info
    table.add_section()
    table.add_row("[bold]Embedding Index[/bold]", "")

    # Check local (Ollama) index - this is the primary/default
    local_sections = VectorStore(name="sections", use_local=True)
    local_summaries = VectorStore(name="summaries", use_local=True)
    local_sections_loaded = local_sections.load()
    local_summaries_loaded = local_summaries.load()

    if local_sections_loaded or local_summaries_loaded:
        table.add_row("  Model", f"[cyan]{DEFAULT_OLLAMA_EMBED_MODEL}[/cyan]")
        if local_sections_loaded:
            sec_val = f"[cyan]{local_sections.size:,}[/cyan] ({local_sections.dimension}d)"
            table.add_row("  Sections", sec_val)
        if local_summaries_loaded:
            sum_val = f"[cyan]{local_summaries.size:,}[/cyan] ({local_summaries.dimension}d)"
            table.add_row("  Summaries", sum_val)
    else:
        table.add_row("  [dim]No index[/dim]", "[yellow]Run 'yt-rag embed'[/yellow]")

    console.print(table)
    db.close()


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@app.command()
def transcript(
    video_id: str = typer.Argument(..., help="Video ID to export"),
    output: Path = typer.Option(None, "-o", "--output", help="Output file path"),
):
    """Export a single video's transcript to a text file."""
    db = get_db()

    video = db.get_video(video_id)
    if not video:
        console.print(f"[red]Video not found: {video_id}[/red]")
        db.close()
        raise typer.Exit(1)

    if video.transcript_status != "fetched":
        console.print(f"[red]Transcript not available (status: {video.transcript_status})[/red]")
        db.close()
        raise typer.Exit(1)

    segments = db.get_segments(video_id)
    if not segments:
        console.print("[red]No transcript segments found[/red]")
        db.close()
        raise typer.Exit(1)

    # Default output path
    if output is None:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video.title)[:50]
        output = Path(tempfile.gettempdir()) / f"{video_id}_{safe_title}.txt"

    with open(output, "w") as f:
        f.write(f"Title: {video.title}\n")
        f.write(f"URL: {video.url}\n")
        f.write(f"Video ID: {video_id}\n")
        f.write("-" * 60 + "\n\n")
        for seg in segments:
            timestamp = format_timestamp(seg.start_time)
            f.write(f"[{timestamp}] {seg.text}\n")

    console.print(f"[green]✓[/green] Exported transcript to {output}")
    db.close()


@app.command("process-transcript")
def process_transcript(
    video_id: str = typer.Argument(None, help="Video ID to process (or all if omitted)"),
    limit: int = typer.Option(None, "-l", "--limit", help="Max videos to process"),
    sectionize_only: bool = typer.Option(False, "--sectionize", help="Only run sectionization"),
    summarize_only: bool = typer.Option(False, "--summarize", help="Only run summarization"),
    model: str = typer.Option(None, "-m", "--model", help="Override default model"),
    openai: bool = typer.Option(False, "--openai", help="Use OpenAI API instead of local Ollama"),
    force: bool = typer.Option(False, "--force", help="Re-process even if already done"),
):
    """Process videos: sectionize (using YouTube chapters) and summarize.

    By default uses local Ollama. Use --openai to use OpenAI API.
    Use --model to override the default model for either backend.

    Sectionizing uses YouTube chapters when available. For videos without
    chapters, LLM-generated titles are created for time-based chunks.
    """
    db = get_db()

    # Determine which videos to process
    if video_id:
        video = db.get_video(video_id)
        if not video:
            console.print(f"[red]Video not found: {video_id}[/red]")
            db.close()
            raise typer.Exit(1)
        if video.transcript_status != "fetched":
            console.print(f"[red]Transcript not available (status: {video.transcript_status})[/]")
            db.close()
            raise typer.Exit(1)
        videos = [video]
    else:
        videos = db.list_videos(status="fetched")
        if limit:
            videos = videos[:limit]

    if not videos:
        console.print("[yellow]No videos to process[/yellow]")
        db.close()
        return

    # Determine steps to run
    do_sectionize = not summarize_only
    do_summarize = not sectionize_only

    sectionized = 0
    chapters_used = 0
    time_chunks_used = 0
    summarized = 0
    errors = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing", total=len(videos))

        for video in videos:
            title = video.title[:35] + "..." if len(video.title) > 35 else video.title

            # Sectionize
            if do_sectionize:
                existing_sections = db.get_sections(video.id)
                if not existing_sections or force:
                    try:
                        progress.update(task, description=f"Sectionizing: {title}")
                        result = sectionize_video(
                            video.id,
                            db,
                            generate_titles=True,
                            model=model,
                            use_openai=openai,
                        )
                        sectionized += 1
                        if result.method == "chapters":
                            chapters_used += 1
                        else:
                            time_chunks_used += 1
                    except Exception as e:
                        console.print(f"\n[red]Sectionize error[/red] {video.id}: {e}")
                        errors += 1
                        progress.advance(task)
                        continue

            # Summarize
            if do_summarize:
                existing_summary = db.get_summary(video.id)
                if not existing_summary or force:
                    try:
                        progress.update(task, description=f"Summarizing: {title}")
                        summarize_video(video.id, db, model=model, use_openai=openai)
                        summarized += 1
                    except Exception as e:
                        console.print(f"\n[red]Summarize error[/red] {video.id}: {e}")
                        errors += 1

            progress.advance(task)

    # Summary message
    parts = [f"[green]✓[/green] Processed {len(videos)} videos:"]
    if sectionized > 0:
        chapters_msg = f"{chapters_used} chapters, {time_chunks_used} time-based"
        parts.append(f"{sectionized} sectionized ({chapters_msg})")
    if summarized > 0:
        parts.append(f"{summarized} summarized")
    if errors > 0:
        parts.append(f"{errors} errors")

    console.print(" ".join(parts))
    db.close()


@app.command()
def embed(
    video_id: str = typer.Argument(None, help="Video ID to embed (or all if omitted)"),
    model: str = typer.Option(None, "-m", "--model", help="Embedding model"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild entire index"),
    force: bool = typer.Option(False, "--force", help="Re-embed existing sections"),
    summaries: bool = typer.Option(True, "--summaries/--no-summaries", help="Embed summaries"),
    openai: bool = typer.Option(False, "--openai", help="Use OpenAI instead of Ollama"),
):
    """Embed sections and summaries into FAISS vector indexes for search.

    By default uses local Ollama embeddings (mxbai-embed-large).
    Use --openai to use OpenAI embeddings (text-embedding-3-small).

    Local and OpenAI indexes are stored separately, so you can switch between them.
    """
    from .config import (
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_OLLAMA_EMBED_MODEL,
        FAISS_DIR,
        FAISS_LOCAL_DIR,
        get_embed_batch_size,
    )
    from .openai_client import check_ollama_running

    use_local = not openai
    backend_name = "local (Ollama)" if use_local else "OpenAI"

    # Check Ollama is running if using local embeddings
    if use_local and not check_ollama_running():
        console.print("[red]Error: Ollama is not running[/red]")
        console.print("Start it with: sudo systemctl start ollama")
        console.print("Or use --openai to use OpenAI embeddings")
        raise typer.Exit(1)

    db = get_db()

    # Show configuration info
    actual_model = model or (DEFAULT_OLLAMA_EMBED_MODEL if use_local else DEFAULT_EMBEDDING_MODEL)
    index_dir = FAISS_LOCAL_DIR if use_local else FAISS_DIR
    batch_size = get_embed_batch_size() if use_local else 100

    console.print(f"[dim]Backend: {backend_name}[/dim]")
    console.print(f"[dim]Model: {actual_model}[/dim]")
    console.print(f"[dim]Batch size: {batch_size}[/dim]")
    console.print(f"[dim]Index dir: {index_dir}[/dim]")

    if video_id:
        # Embed single video
        video = db.get_video(video_id)
        if not video:
            console.print(f"[red]Video not found: {video_id}[/red]")
            db.close()
            raise typer.Exit(1)

        sections = db.get_sections(video_id)
        if not sections:
            console.print(f"[red]No sections found for video {video_id}[/red]")
            console.print("Run 'yt-rag process' first to create sections.")
            db.close()
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Embedding {len(sections)} sections...", total=None)
            result = embed_video(video_id, db, model, force, use_local=use_local)

        console.print(
            f"[green]✓[/green] Embedded {result.items_embedded} sections "
            f"({result.tokens_used} tokens)"
        )
    else:
        # Embed all sections
        stats = db.get_stats()
        if stats["sections"] == 0:
            console.print("[yellow]No sections to embed[/yellow]")
            console.print("Run 'yt-rag process' first to create sections.")
            db.close()
            return

        console.print(f"[dim]Total sections in DB: {stats['sections']:,}[/dim]")

        total_tokens = 0

        # Embed sections with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            desc = "Rebuilding sections index..." if rebuild else "Embedding new sections..."
            task = progress.add_task(desc, total=None)

            def update_progress(embedded: int, total: int) -> None:
                if progress.tasks[task].total is None:
                    progress.update(task, total=total)
                progress.update(task, completed=embedded)

            result = embed_all_sections(
                db, model, rebuild=rebuild, use_local=use_local, progress_callback=update_progress
            )
            total_tokens += result.tokens_used

        if result.items_embedded > 0:
            console.print(
                f"[green]✓[/green] Embedded {result.items_embedded} sections "
                f"({result.tokens_used} tokens)"
            )
        else:
            console.print("[green]✓[/green] All sections already embedded")

        # Embed summaries for video-level search
        if summaries and stats["summaries"] > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                if rebuild:
                    progress.add_task("Rebuilding summaries index...", total=None)
                else:
                    progress.add_task("Embedding new summaries...", total=None)

                summary_result = embed_all_summaries(
                    db, model, rebuild=rebuild, use_local=use_local
                )
                total_tokens += summary_result.tokens_used

            if summary_result.items_embedded > 0:
                console.print(
                    f"[green]✓[/green] Embedded {summary_result.items_embedded} video summaries "
                    f"({summary_result.tokens_used} tokens)"
                )
            else:
                console.print("[green]✓[/green] All summaries already embedded")

    # Show index stats
    idx_stats = get_index_stats(use_local=use_local)
    sec_count = idx_stats["sections_vectors"]
    sum_count = idx_stats["summaries_vectors"]
    console.print(f"Index: {sec_count} sections, {sum_count} summaries")
    console.print(f"[dim]Index dir: {idx_stats['index_dir']}[/dim]")

    db.close()


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "-k", "--top-k", help="Number of sources to retrieve"),
    video: str = typer.Option(None, "-v", "--video", help="Filter to specific video ID"),
    channel: str = typer.Option(None, "-c", "--channel", help="Filter to specific channel ID"),
    no_answer: bool = typer.Option(False, "--no-answer", help="Skip answer generation"),
    model: str = typer.Option(None, "-m", "--model", help="Override default model"),
    openai: bool = typer.Option(False, "--openai", help="Use OpenAI API instead of local Ollama"),
):
    """Ask a question about video content using RAG.

    By default uses local Ollama. Use --openai to use OpenAI API.
    Use --model to override the default model for either backend.
    """
    # Determine backend and model
    use_local = not openai
    if model:
        chat_model = model
    else:
        chat_model = DEFAULT_OLLAMA_MODEL if use_local else DEFAULT_CHAT_MODEL
    db = get_db()

    # For list/show queries, skip LLM answer generation (CLI formats the output)
    is_list_query = any(kw in query.lower() for kw in ["show", "list", "find", "top "])
    should_generate_answer = not no_answer and not is_list_query

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Searching...", total=None)
        result = rag_search(
            query=query,
            db=db,
            top_k=top_k,
            video_id=video,
            channel_id=channel,
            generate_answer=should_generate_answer,
            chat_model=chat_model,
            use_local=use_local,
        )

    if not result.hits:
        console.print("[yellow]No relevant content found.[/yellow]")
        db.close()
        return

    # Show LLM answer if generated (for regular questions, not list queries)
    if result.answer:
        console.print("\n[bold]Answer:[/bold]")
        console.print(result.answer)
        console.print()

    # Show results list with rich formatting (no duplicate "Sources" header)
    console.print()
    for i, hit in enumerate(result.hits, 1):
        channel_str = f"[magenta]{hit.channel_name}[/magenta] | " if hit.channel_name else ""
        if hit.published_at:
            date_str = f" [dim]({hit.published_at.strftime('%Y-%m-%d')})[/dim]"
        else:
            date_str = ""
        # Truncate section content for preview (first 150 chars)
        content_preview = hit.section.content[:150].replace("\n", " ").strip()
        if len(hit.section.content) > 150:
            content_preview += "..."

        console.print(f"[cyan]{i}.[/cyan] {channel_str}[bold]{hit.video_title}[/bold]{date_str}")
        console.print(f"   Section: {hit.section.title}")
        console.print(f"   [dim]{content_preview}[/dim]")
        console.print(f"   [link={hit.timestamp_url}]{hit.timestamp_url}[/link]")
        console.print()

    # Show stats
    if should_generate_answer:
        model_info = f" | Model: {chat_model}"
        console.print(
            f"\n[dim]Latency: {result.latency_ms}ms | "
            f"Tokens: {result.tokens_embedding} embed + {result.tokens_chat} chat{model_info}[/dim]"
        )
    else:
        stats = f"Latency: {result.latency_ms}ms | Tokens: {result.tokens_embedding} embed"
        console.print(f"\n[dim]{stats}[/dim]")

    db.close()


@app.command()
def logs(
    limit: int = typer.Option(20, "-n", "--limit", help="Number of logs to show"),
    query_id: str = typer.Option(None, "-q", "--query-id", help="Show specific query"),
):
    """View query logs."""
    db = get_db()

    if query_id:
        # Show specific query
        logs_list = db.get_query_logs(limit=1000)
        log = next((entry for entry in logs_list if entry.id == query_id), None)
        if not log:
            console.print(f"[red]Query not found: {query_id}[/red]")
            db.close()
            raise typer.Exit(1)

        console.print(f"\n[bold]Query ID:[/bold] {log.id}")
        console.print(f"[bold]Query:[/bold] {log.query}")
        console.print(f"[bold]Created:[/bold] {log.created_at}")
        console.print(f"[bold]Latency:[/bold] {log.latency_ms}ms")
        console.print(f"[bold]Tokens:[/bold] {log.tokens_embedding} embed + {log.tokens_chat} chat")

        if log.scope_type:
            console.print(f"[bold]Scope:[/bold] {log.scope_type}={log.scope_id}")

        if log.retrieved_ids:
            console.print(f"\n[bold]Retrieved ({len(log.retrieved_ids)}):[/bold]")
            for i, (rid, score) in enumerate(zip(log.retrieved_ids, log.retrieved_scores or []), 1):
                console.print(f"  {i}. {rid} (score: {score:.3f})")

        if log.answer:
            console.print(f"\n[bold]Answer:[/bold]\n{log.answer}")
    else:
        # List recent logs
        logs_list = db.get_query_logs(limit=limit)

        if not logs_list:
            console.print("[yellow]No query logs found[/yellow]")
            db.close()
            return

        table = Table(title=f"Recent Queries ({len(logs_list)})")
        table.add_column("ID", style="dim")
        table.add_column("Query", max_width=40)
        table.add_column("Hits")
        table.add_column("Latency")
        table.add_column("Time")

        for log in logs_list:
            hits = len(log.retrieved_ids) if log.retrieved_ids else 0
            time_str = log.created_at.strftime("%m-%d %H:%M") if log.created_at else ""
            table.add_row(
                log.id[:8],
                log.query[:40] + ("..." if len(log.query) > 40 else ""),
                str(hits),
                f"{log.latency_ms}ms",
                time_str,
            )

        console.print(table)

    db.close()


@app.command()
def feedback(
    query_id: str = typer.Argument(..., help="Query ID to rate"),
    helpful: bool = typer.Option(None, "--helpful/--not-helpful", help="Was answer helpful?"),
    rating: int = typer.Option(None, "-r", "--rating", help="Source quality 1-5"),
    comment: str = typer.Option(None, "-c", "--comment", help="Optional comment"),
):
    """Add feedback for a query."""
    db = get_db()

    # Verify query exists
    logs_list = db.get_query_logs(limit=1000)
    log = next((entry for entry in logs_list if entry.id.startswith(query_id)), None)
    if not log:
        console.print(f"[red]Query not found: {query_id}[/red]")
        db.close()
        raise typer.Exit(1)

    if rating and (rating < 1 or rating > 5):
        console.print("[red]Rating must be 1-5[/red]")
        db.close()
        raise typer.Exit(1)

    add_feedback(
        query_id=log.id,
        helpful=helpful,
        source_rating=rating,
        comment=comment,
        db=db,
    )

    console.print(f"[green]✓[/green] Feedback saved for query {log.id[:8]}")
    db.close()


@app.command("eval")
def eval_cmd(
    show_failures: bool = typer.Option(False, "--failures", help="Show failed tests only"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show detailed results"),
):
    """Run evaluation benchmark."""
    db = get_db()

    tests = db.get_test_cases()
    if not tests:
        console.print("[yellow]No test cases found[/yellow]")
        console.print('Add test cases with: yt-rag test-add "query" --videos=id1,id2')
        db.close()
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Running {len(tests)} tests...", total=None)
        result = run_benchmark(db)

    # Summary
    console.print("\n[bold]Benchmark Results[/bold]")
    console.print(f"  Tests: {result.passed}/{result.total_tests} passed")
    console.print(f"  Precision@5: {result.avg_precision:.1%}")
    console.print(f"  Recall: {result.avg_recall:.1%}")
    console.print(f"  MRR: {result.avg_mrr:.2f}")
    console.print(f"  Keyword Match: {result.avg_keyword_match:.1%}")
    console.print(f"  Avg Latency: {result.avg_latency_ms:.0f}ms")

    # Details
    if verbose or show_failures:
        console.print()
        for r in result.results:
            if show_failures and r.passed:
                continue

            status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            console.print(f"\n{status} [{r.test_id}] {r.query}")
            console.print(f"  P@5={r.precision_at_k:.1%} R={r.recall:.1%} MRR={r.mrr:.2f}")
            if r.expected_video_ids:
                console.print(f"  Expected: {r.expected_video_ids}")
                console.print(f"  Got: {r.retrieved_video_ids[:5]}")

    db.close()


@app.command("test-add")
def test_add(
    query: str = typer.Argument(..., help="Test query"),
    videos: str = typer.Option(None, "--videos", help="Expected video IDs (comma-separated)"),
    keywords: str = typer.Option(None, "--keywords", help="Expected keywords (comma-separated)"),
    answer: str = typer.Option(None, "--answer", help="Reference answer"),
):
    """Add a test case for evaluation."""
    db = get_db()

    video_ids = [v.strip() for v in videos.split(",")] if videos else None
    keyword_list = [k.strip() for k in keywords.split(",")] if keywords else None

    test = add_test_case(
        query=query,
        expected_video_ids=video_ids,
        expected_keywords=keyword_list,
        reference_answer=answer,
        db=db,
    )

    console.print(f"[green]✓[/green] Added test case: {test.id}")
    db.close()


@app.command("test-list")
def test_list():
    """List all test cases."""
    db = get_db()
    tests = db.get_test_cases()

    if not tests:
        console.print("[yellow]No test cases[/yellow]")
        db.close()
        return

    table = Table(title=f"Test Cases ({len(tests)})")
    table.add_column("ID")
    table.add_column("Query", max_width=50)
    table.add_column("Videos")
    table.add_column("Keywords")

    for t in tests:
        videos = ",".join(t.expected_video_ids[:2]) if t.expected_video_ids else "-"
        keywords = ",".join(t.expected_keywords[:3]) if t.expected_keywords else "-"
        table.add_row(t.id, t.query[:50], videos, keywords)

    console.print(table)
    db.close()


@app.command("test")
def test_benchmark(
    data_file: str = typer.Option(
        None,
        "--data",
        "-d",
        help="Path to JSON test data file (default: tests/data/benchmark.json)",
    ),
    output_file: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results to JSON file (auto-named if not specified)",
    ),
    model: str = typer.Option(
        None, "-m", "--model", help="Override default LLM model for RAG pipeline"
    ),
    openai: bool = typer.Option(False, "--openai", help="Use OpenAI API instead of local Ollama"),
    validate_openai: bool = typer.Option(
        False, "--validate-openai", help="Also validate with OpenAI (compares validators)"
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Show all results, not just failures"
    ),
):
    """Benchmark full RAG pipeline: classification, retrieval, and answer quality.

    By default uses local Ollama. Use --openai to use OpenAI API.
    Use --model to override the default model for either backend.

    Validation includes:
    - Keyword matching: expected keywords found in answer
    - LLM validation: Uses same backend as pipeline (and GPT-4o if --validate-openai)

    Use --validate-openai to compare local vs GPT-4o as validators side-by-side.
    """
    import json
    from pathlib import Path

    # Find test data file
    if data_file:
        data_path = Path(data_file)
    else:
        # Look for default test data relative to package
        pkg_dir = Path(__file__).parent.parent.parent  # src/yt_rag -> src -> project root
        data_path = pkg_dir / "tests" / "data" / "benchmark.json"

    if not data_path.exists():
        console.print(f"[red]Error: Test data file not found: {data_path}[/red]")
        console.print(
            "Create a JSON file with test_cases array containing "
            "query/expected_type/expected_keywords"
        )
        raise typer.Exit(1)

    # Load test data
    with open(data_path) as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    if not test_cases:
        console.print("[yellow]No test cases found in data file[/yellow]")
        raise typer.Exit(1)

    # Auto-generate output path
    if output_file:
        output_path = Path(output_file)
    else:
        base_name = data_path.name.replace("benchmark_generated", "benchmark_results")
        output_path = data_path.parent / base_name

    _run_full_pipeline_test(
        test_cases,
        data_path,
        output_path,
        verbose,
        model=model,
        use_openai=openai,
        validate_openai=validate_openai,
    )


# LLM validation prompt for judging answer quality
VALIDATION_PROMPT = """You are evaluating a RAG system's answer quality.

Question: {query}
Expected to mention: {expected_keywords}
RAG Answer: {answer}

Evaluate if the answer correctly addresses the question. Consider:
1. Does the answer mention the key entity/topic from the question?
2. Is the answer relevant and informative?
3. Does it correctly use information (not hallucinating)?

Respond with JSON only:
{{"pass": true/false, "reason": "brief explanation"}}"""


def _validate_with_llm(
    query: str,
    answer: str,
    expected_keywords: list,
    use_openai: bool = False,
    model: str | None = None,
) -> dict:
    """Use LLM to validate answer quality.

    Args:
        query: The query that was asked
        answer: The RAG answer to validate
        expected_keywords: Keywords expected in the answer
        use_openai: Use OpenAI API instead of local Ollama
        model: Override default model (if None, uses default based on use_openai)

    Returns:
        dict with 'pass' (bool) and 'reason' (str)
    """
    import json as json_module

    from .config import DEFAULT_CHAT_MODEL, DEFAULT_OLLAMA_MODEL
    from .openai_client import chat_completion, ollama_chat_completion

    if not answer or not answer.strip():
        return {"pass": False, "reason": "Empty answer"}

    prompt = VALIDATION_PROMPT.format(
        query=query,
        expected_keywords=", ".join(expected_keywords) if expected_keywords else "N/A",
        answer=answer[:500],  # Truncate long answers
    )

    try:
        if use_openai:
            actual_model = model if model else DEFAULT_CHAT_MODEL
            response = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=actual_model,
                temperature=0.0,
                max_tokens=100,
            )
        else:
            actual_model = model if model else DEFAULT_OLLAMA_MODEL
            response = ollama_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=actual_model,
                temperature=0.0,
            )

        content = response.content.strip()
        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json_module.loads(content)
        return {
            "pass": bool(result.get("pass", False)),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        return {"pass": False, "reason": f"Validation error: {e}"}


def _run_full_pipeline_test(
    test_cases: list,
    data_path,
    output_path,
    verbose: bool,
    model: str | None = None,
    use_openai: bool = False,
    validate_openai: bool = False,
):
    """Run full pipeline test with timing breakdown.

    By default uses local Ollama. Set use_openai=True for OpenAI.
    Validation uses both keyword matching and LLM-based judging.
    If validate_openai=True, also validates with GPT-4o for comparison.

    Args:
        test_cases: List of test case dicts
        data_path: Path to input data file
        output_path: Path for output results
        verbose: Show detailed output
        model: Override default LLM model
        use_openai: Use OpenAI API instead of local Ollama
        validate_openai: Also validate with OpenAI for comparison
    """
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    from .config import (
        DEFAULT_CHAT_MODEL,
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_OLLAMA_EMBED_MODEL,
        DEFAULT_OLLAMA_MODEL,
    )
    from .openai_client import (
        chat_completion,
        embed_text,
        ollama_chat_completion,
        ollama_embed_text,
    )
    from .search import (
        RAG_SYSTEM_PROMPT,
        RAG_USER_PROMPT,
        analyze_query_with_llm,
        format_context,
        get_sections_store,
    )

    # Determine models based on flags
    use_local = not use_openai
    if use_local:
        chat_model = model if model else DEFAULT_OLLAMA_MODEL
        embed_model = DEFAULT_OLLAMA_EMBED_MODEL
        backend_name = f"local Ollama ({chat_model})"
    else:
        chat_model = model if model else DEFAULT_CHAT_MODEL
        embed_model = DEFAULT_EMBEDDING_MODEL
        backend_name = f"OpenAI ({chat_model})"

    console.print(f"[bold]Running {len(test_cases)} full pipeline tests[/bold]")
    console.print(f"Pipeline: {backend_name}")
    console.print(f"Validators: {backend_name}" + (" + GPT-4o" if validate_openai else ""))
    console.print(f"Data file: {data_path}")
    console.print(f"Output file: {output_path}\n")

    db = get_db()
    store = get_sections_store(use_local=use_local)

    if store.size == 0:
        console.print("[red]No vectors indexed. Run 'yt-rag embed' first.[/red]")
        db.close()
        raise typer.Exit(1)

    # Aggregate timing stats
    timings = {
        "parallel": [],
        "search": [],
        "filter": [],
        "answer": [],
        "validate": [],
        "total": [],
    }

    # Evaluation results
    results = []
    classification_correct = 0
    keyword_hits = 0
    keyword_total = 0
    main_validation_pass = 0
    gpt_validation_pass = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Testing...", total=len(test_cases))

        for case in test_cases:
            query = case["query"]
            expected_type = case.get("expected_type")
            expected_keywords = case.get("expected_keywords", [])
            t_start = time.time()

            # Handle META queries specially - no search/LLM needed
            if expected_type == "meta":
                from .search import classify_query

                t_classify = time.time()
                got_type = classify_query(query).value
                type_passed = got_type == "meta"
                if type_passed:
                    classification_correct += 1

                # For meta queries, the "answer" is the stats from DB
                stats = db.get_stats()
                channels = db.list_channels()
                answer = (
                    f"Library contains {stats.get('videos_fetched', 0):,} videos "
                    f"across {len(channels)} channels."
                )

                # META queries always pass validation (no keywords to check)
                result_entry = {
                    "query": query,
                    "expected_type": expected_type,
                    "got_type": got_type,
                    "type_passed": type_passed,
                    "expected_keywords": [],
                    "keywords_found": [],
                    "keywords_missing": [],
                    "answer": answer,
                    "validation_main": {"pass": True, "reason": "Meta query - stats returned"},
                    "validation_model": chat_model,
                }
                if validate_openai:
                    result_entry["validation_gpt4o"] = {
                        "pass": True,
                        "reason": "Meta query - stats returned",
                    }
                    gpt_validation_pass += 1
                main_validation_pass += 1
                results.append(result_entry)

                # Minimal timing entries
                timings["parallel"].append(int((time.time() - t_classify) * 1000))
                timings["search"].append(0)
                timings["filter"].append(0)
                timings["answer"].append(0)
                timings["validate"].append(0)
                timings["total"].append(int((time.time() - t_start) * 1000))
                progress.update(task, advance=1)
                continue

            # Stage 1: Parallel parsing and embedding
            t_parallel_start = time.time()

            def do_parse():
                return analyze_query_with_llm(query, use_local=use_local, model=chat_model)

            def do_embed():
                if use_local:
                    return ollama_embed_text(query, embed_model)
                else:
                    return embed_text(query, embed_model)

            with ThreadPoolExecutor(max_workers=2) as executor:
                parse_future = executor.submit(do_parse)
                embed_future = executor.submit(do_embed)
                analysis = parse_future.result()
                embed_result = embed_future.result()

            t_parallel_end = time.time()
            timings["parallel"].append(int((t_parallel_end - t_parallel_start) * 1000))

            # Classification check
            got_type = analysis.query_type.value if analysis else "ERROR"
            type_passed = got_type == expected_type if expected_type else True
            if type_passed and expected_type:
                classification_correct += 1

            # Stage 2: Vector search
            t_search_start = time.time()
            search_results = store.search(query_embedding=embed_result.embedding, top_k=50)
            t_search_end = time.time()
            timings["search"].append(int((t_search_end - t_search_start) * 1000))

            # Stage 3: Keyword filtering
            t_filter_start = time.time()
            if analysis and analysis.keywords:
                keywords = set(analysis.keywords)
                filtered = []
                for result in search_results:
                    section = db.get_section(result.id)
                    if not section:
                        continue
                    video = db.get_video(section.video_id)
                    if not video:
                        continue
                    all_text = f"{video.title} {section.title} {section.content}".lower()
                    if any(kw in all_text for kw in keywords):
                        filtered.append(result)
                search_results = filtered[:10]
            else:
                search_results = search_results[:10]
            t_filter_end = time.time()
            timings["filter"].append(int((t_filter_end - t_filter_start) * 1000))

            # Stage 4: Answer generation (always local)
            t_answer_start = time.time()
            hits = []
            for r in search_results[:5]:
                section = db.get_section(r.id)
                if section:
                    video = db.get_video(section.video_id)
                    if video:
                        from .search import SearchHit

                        hits.append(
                            SearchHit(
                                section=section,
                                video_id=video.id,
                                video_title=video.title,
                                video_url=video.url,
                                channel_id=video.channel_id,
                                channel_name=None,
                                host=None,
                                tags=None,
                                score=r.score,
                                timestamp_url=f"https://youtube.com/watch?v={video.id}",
                            )
                        )

            answer = ""
            if hits:
                context = format_context(hits, db=db)
                user_content = RAG_USER_PROMPT.format(context=context, question=query)
                messages = [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                if use_local:
                    response = ollama_chat_completion(
                        messages=messages, model=chat_model, temperature=0.1
                    )
                else:
                    response = chat_completion(messages=messages, model=chat_model, temperature=0.1)
                answer = response.content

            t_answer_end = time.time()
            timings["answer"].append(int((t_answer_end - t_answer_start) * 1000))

            # Stage 5: Validation
            t_validate_start = time.time()

            # Keyword validation
            answer_lower = answer.lower()
            keywords_found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            keywords_missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
            keyword_hits += len(keywords_found)
            keyword_total += len(expected_keywords)

            # LLM validation (same backend as pipeline)
            main_validation = _validate_with_llm(
                query, answer, expected_keywords, use_openai=use_openai, model=chat_model
            )
            if main_validation["pass"]:
                main_validation_pass += 1

            # LLM validation (GPT-4o) if requested - always uses OpenAI
            gpt_result = None
            if validate_openai:
                gpt_result = _validate_with_llm(query, answer, expected_keywords, use_openai=True)
                if gpt_result["pass"]:
                    gpt_validation_pass += 1

            t_validate_end = time.time()
            timings["validate"].append(int((t_validate_end - t_validate_start) * 1000))

            t_end = time.time()
            timings["total"].append(int((t_end - t_start) * 1000))

            # Store result
            result_entry = {
                "query": query,
                "expected_type": expected_type,
                "got_type": got_type,
                "type_passed": type_passed,
                "expected_keywords": expected_keywords,
                "keywords_found": keywords_found,
                "keywords_missing": keywords_missing,
                "answer": answer[:500],
                "validation_main": main_validation,
                "validation_model": chat_model,
            }
            if gpt_result:
                result_entry["validation_gpt4o"] = gpt_result
            results.append(result_entry)

            progress.update(task, advance=1)

    db.close()

    # Print timing summary
    console.print()
    console.print("[bold]Timing Breakdown (avg ms)[/bold]")

    timing_table = Table()
    timing_table.add_column("Stage")
    timing_table.add_column("Avg (ms)", justify="right")
    timing_table.add_column("Min", justify="right")
    timing_table.add_column("Max", justify="right")

    for stage, values in timings.items():
        if not values:
            continue
        timing_table.add_row(
            stage.capitalize(),
            f"{sum(values) / len(values):.0f}",
            f"{min(values)}",
            f"{max(values)}",
        )
    console.print(timing_table)

    # Quality metrics
    console.print()
    console.print("[bold]Quality Metrics[/bold]")

    type_cases = [r for r in results if r["expected_type"]]
    if type_cases:
        pct = classification_correct / len(type_cases) * 100
        console.print(f"  Classification: {classification_correct}/{len(type_cases)} ({pct:.1f}%)")

    if keyword_total > 0:
        pct = keyword_hits / keyword_total * 100
        console.print(f"  Keyword Match: {keyword_hits}/{keyword_total} ({pct:.1f}%)")

    # LLM Validation comparison
    console.print()
    console.print("[bold]LLM Validation Results[/bold]")

    val_table = Table()
    val_table.add_column("Validator")
    val_table.add_column("Pass", justify="right")
    val_table.add_column("Fail", justify="right")
    val_table.add_column("Rate", justify="right")

    total = len(test_cases)
    validator_name = chat_model if use_local else f"OpenAI ({chat_model})"
    val_table.add_row(
        validator_name,
        str(main_validation_pass),
        str(total - main_validation_pass),
        f"{main_validation_pass / total * 100:.1f}%",
    )

    if validate_openai:
        val_table.add_row(
            "GPT-4o (comparison)",
            str(gpt_validation_pass),
            str(total - gpt_validation_pass),
            f"{gpt_validation_pass / total * 100:.1f}%",
        )

        # Agreement analysis
        agree = sum(
            1
            for r in results
            if r["validation_main"]["pass"] == r.get("validation_gpt4o", {}).get("pass")
        )
        console.print(val_table)
        console.print(f"\n  Agreement: {agree}/{total} ({agree / total * 100:.1f}%)")

        # Disagreements
        main_only = [
            r
            for r in results
            if r["validation_main"]["pass"] and not r.get("validation_gpt4o", {}).get("pass")
        ]
        gpt_only = [
            r
            for r in results
            if not r["validation_main"]["pass"] and r.get("validation_gpt4o", {}).get("pass")
        ]
        console.print(f"  Main pass, GPT fail: {len(main_only)}")
        console.print(f"  GPT pass, Main fail: {len(gpt_only)}")
    else:
        console.print(val_table)

    # Show failures if verbose
    if verbose:
        failures = [r for r in results if not r["validation_main"]["pass"]]
        if failures:
            console.print(f"\n[yellow]Validation Failures ({len(failures)}):[/yellow]")
            for r in failures[:10]:
                console.print(f"  {r['query'][:50]}")
                console.print(f"    Reason: {r['validation_main']['reason']}")

    # Save results
    output_data = {
        "source_file": str(data_path),
        "pipeline": backend_name,
        "pipeline_model": chat_model,
        "validators": [chat_model] + (["gpt-4o"] if validate_openai else []),
        "total_tests": len(test_cases),
        "summary": {
            "classification": {
                "correct": classification_correct,
                "total": len(type_cases),
                "accuracy": classification_correct / len(type_cases) * 100 if type_cases else 0,
            },
            "keyword_match": {
                "hits": keyword_hits,
                "total": keyword_total,
                "accuracy": keyword_hits / keyword_total * 100 if keyword_total else 0,
            },
            "llm_validation": {
                "main": {
                    "model": chat_model,
                    "pass": main_validation_pass,
                    "rate": main_validation_pass / total * 100,
                },
            },
            "timing_avg_ms": {k: sum(v) / len(v) if v else 0 for k, v in timings.items()},
        },
        "results": results,
    }

    if validate_openai:
        output_data["summary"]["llm_validation"]["gpt4o"] = {
            "pass": gpt_validation_pass,
            "rate": gpt_validation_pass / total * 100,
        }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")


def _format_duration(seconds: int | None) -> str:
    """Format duration as HH:MM:SS or MM:SS."""
    if seconds is None:
        return "-"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _parse_date(date_str: str):
    """Parse date string in YYYY-MM-DD format."""
    from datetime import datetime

    return datetime.strptime(date_str, "%Y-%m-%d")


def _parse_duration(duration_str: str) -> int:
    """Parse duration string like '1h30m', '45m', '90s', or just minutes."""
    import re

    duration_str = duration_str.strip().lower()

    # Try HH:MM:SS or MM:SS format
    if ":" in duration_str:
        parts = duration_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])

    # Try 1h30m format
    hours = 0
    minutes = 0
    seconds = 0

    h_match = re.search(r"(\d+)h", duration_str)
    m_match = re.search(r"(\d+)m", duration_str)
    s_match = re.search(r"(\d+)s", duration_str)

    if h_match:
        hours = int(h_match.group(1))
    if m_match:
        minutes = int(m_match.group(1))
    if s_match:
        seconds = int(s_match.group(1))

    if hours or minutes or seconds:
        return hours * 3600 + minutes * 60 + seconds

    # Fallback: treat as minutes
    return int(duration_str) * 60


@app.command()
def videos(
    channel: str = typer.Option(None, "-c", "--channel", help="Filter by channel ID"),
    status: str = typer.Option(None, "--status", help="Filter by status"),
    after: str = typer.Option(None, "--after", help="Published after (YYYY-MM-DD)"),
    before: str = typer.Option(None, "--before", help="Published before (YYYY-MM-DD)"),
    min_duration: str = typer.Option(
        None, "--min-duration", help="Min duration (e.g., 30m, 1h, 1:30:00)"
    ),
    max_duration: str = typer.Option(
        None, "--max-duration", help="Max duration (e.g., 30m, 1h, 1:30:00)"
    ),
    host: str = typer.Option(None, "--host", help="Filter by host name"),
    guest: str = typer.Option(None, "--guest", help="Filter by guest name"),
    min_sections: int = typer.Option(None, "--min-sections", help="Min section count"),
    max_sections: int = typer.Option(None, "--max-sections", help="Max section count"),
    title: str = typer.Option(None, "-t", "--title", help="Title contains"),
    order: str = typer.Option(
        "date", "-o", "--order", help="Order by: date, duration, title, sections"
    ),
    asc: bool = typer.Option(False, "--asc", help="Sort ascending"),
    limit: int = typer.Option(50, "-n", "--limit", help="Max results"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show more details"),
):
    """Search and filter videos by metadata.

    Examples:
        yt-rag videos --after 2024-01-01
        yt-rag videos --min-duration 30m --max-duration 2h
        yt-rag videos --host "Doug" --order duration
        yt-rag videos --min-sections 10 --order sections
        yt-rag videos --title "review" --after 2024-06-01
    """
    db = get_db()

    # Parse date filters
    after_dt = _parse_date(after) if after else None
    before_dt = _parse_date(before) if before else None

    # Parse duration filters
    min_dur = _parse_duration(min_duration) if min_duration else None
    max_dur = _parse_duration(max_duration) if max_duration else None

    # Map order parameter
    order_map = {
        "date": "published_at",
        "duration": "duration_seconds",
        "title": "title",
        "sections": "section_count",
    }
    order_by = order_map.get(order, "published_at")

    results = db.search_videos(
        channel_id=channel,
        status=status,
        after=after_dt,
        before=before_dt,
        min_duration=min_dur,
        max_duration=max_dur,
        host=host,
        guest=guest,
        min_sections=min_sections,
        max_sections=max_sections,
        title_contains=title,
        order_by=order_by,
        order_desc=not asc,
        limit=limit,
    )

    if not results:
        console.print("[yellow]No videos found matching filters[/yellow]")
        db.close()
        return

    table = Table(title=f"Videos ({len(results)} results)")
    table.add_column("ID", style="dim")
    table.add_column("Title", max_width=45)
    table.add_column("Duration", justify="right")
    table.add_column("Sections", justify="right")
    table.add_column("Published")

    if verbose:
        table.add_column("Host")
        table.add_column("Guests")

    for video, section_count in results:
        duration_str = _format_duration(video.duration_seconds)
        pub_date = video.published_at.strftime("%Y-%m-%d") if video.published_at else "-"
        title_display = video.title[:45] + "..." if len(video.title) > 45 else video.title

        if verbose:
            host_str = video.host or "-"
            guests_str = ", ".join(video.guests[:2]) if video.guests else "-"
            if video.guests and len(video.guests) > 2:
                guests_str += f" +{len(video.guests) - 2}"
            table.add_row(
                video.id,
                title_display,
                duration_str,
                str(section_count),
                pub_date,
                host_str,
                guests_str,
            )
        else:
            table.add_row(
                video.id,
                title_display,
                duration_str,
                str(section_count),
                pub_date,
            )

    console.print(table)
    db.close()


@app.command("refresh-meta")
def refresh_meta(
    video_only: bool = typer.Option(False, "--video", help="Refresh only video metadata"),
    channel_only: bool = typer.Option(False, "--channel", help="Refresh only channel metadata"),
    limit: int = typer.Option(None, "-l", "--limit", help="Max videos to refresh"),
    force: bool = typer.Option(False, "--force", help="Refresh even if metadata exists"),
):
    """Refresh metadata from YouTube.

    By default refreshes both videos and channels.
    Use --video or --channel to refresh only one type.
    """
    db = get_db()

    do_videos = not channel_only
    do_channels = not video_only

    work_items: list[tuple[str, Channel | Video]] = []

    if do_channels:
        channels = db.list_channels()
        if not force:
            channels = [c for c in channels if not c.description]
        work_items.extend([("channel", c) for c in channels])

    if do_videos:
        videos = db.list_videos(status="fetched")
        if not force:
            videos = [v for v in videos if not v.description]
        if limit:
            videos = videos[:limit]
        work_items.extend([("video", v) for v in videos])

    if not work_items:
        console.print("[green]✓[/green] All metadata up to date")
        db.close()
        return

    channel_count = sum(1 for t, _ in work_items if t == "channel")
    video_count = len(work_items) - channel_count
    console.print(f"Refreshing {channel_count} channels + {video_count} videos")

    result = asyncio.run(refresh_metadata_async(work_items, db))

    parts = []
    if result.channels_updated or result.channels_errors:
        parts.append(f"{result.channels_updated} channels")
    if result.videos_updated or result.videos_errors:
        parts.append(f"{result.videos_updated} videos")
    total_errors = result.channels_errors + result.videos_errors
    error_msg = f", {total_errors} errors" if total_errors else ""
    console.print(f"[green]✓[/green] Updated {', '.join(parts)}{error_msg}")

    db.close()


@app.command()
def version():
    """Show version."""
    console.print(f"yt-rag {__version__}")


@app.command()
def keywords(
    limit: int = typer.Option(10, "-n", "--limit", help="Number of videos to analyze"),
    top_k: int = typer.Option(50, "-k", "--top-k", help="Top keywords to show"),
    channel: str = typer.Option(None, "-c", "--channel", help="Filter by channel ID"),
    save: bool = typer.Option(False, "--save", help="Save keywords to database"),
):
    """Extract and analyze keywords from video transcripts."""
    from .keywords import extract_keywords_from_videos, suggest_synonyms_for_keyword

    db = get_db()
    conn = db.connect()

    # Get videos to analyze
    if channel:
        rows = conn.execute(
            """
            SELECT v.id, v.title FROM videos v
            JOIN sections s ON v.id = s.video_id
            WHERE v.transcript_status = 'fetched' AND v.channel_id = ?
            GROUP BY v.id
            LIMIT ?
            """,
            (channel, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT v.id, v.title FROM videos v
            JOIN sections s ON v.id = s.video_id
            WHERE v.transcript_status = 'fetched'
            GROUP BY v.id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    if not rows:
        console.print("[yellow]No videos with transcripts found.[/yellow]")
        db.close()
        return

    video_ids = [row[0] for row in rows]
    console.print(f"Analyzing {len(video_ids)} videos...")

    keywords_list = extract_keywords_from_videos(db, video_ids, min_total_frequency=5)

    console.print(f"\n[bold]Top {min(top_k, len(keywords_list))} keywords:[/bold]\n")

    table = Table()
    table.add_column("Freq", justify="right", style="cyan")
    table.add_column("Videos", justify="right")
    table.add_column("Keyword", style="green")
    table.add_column("Suggested Synonyms", style="dim")

    for kw in keywords_list[:top_k]:
        synonyms = suggest_synonyms_for_keyword(kw.keyword)
        syn_str = ", ".join(synonyms[:4]) if synonyms else "-"
        table.add_row(str(kw.frequency), str(kw.video_count), kw.keyword, syn_str)

    console.print(table)

    if save:
        for kw in keywords_list:
            db.upsert_keyword(kw.keyword, kw.frequency, kw.video_count)
        console.print(f"\n[green]✓[/green] Saved {len(keywords_list)} keywords to database")

    db.close()


@app.command()
def synonyms(
    action: str = typer.Argument("list", help="Action: list, generate, approve, reject, remove"),
    keywords: list[str] = typer.Argument(None, help="Keywords to remove (for 'remove' action)"),
    keyword: str = typer.Option(None, "-k", "--keyword", help="Keyword to work with"),
    synonym: str = typer.Option(None, "-s", "--synonym", help="Synonym to add/approve/reject"),
    pending: bool = typer.Option(False, "--pending", help="Show pending synonyms only"),
    limit: int = typer.Option(10, "-n", "--limit", help="Number of keywords to process"),
    model: str = typer.Option(
        None, "-m", "--model", help="Override default LLM model (for generate)"
    ),
    openai: bool = typer.Option(
        False, "--openai", help="Use OpenAI API instead of local Ollama (for generate)"
    ),
):
    """Manage synonym mappings for search boosting.

    Actions:
      list     - List current synonyms
      generate - Generate synonym suggestions using LLM
      approve  - Approve a synonym (keyword + synonym required)
      reject   - Reject a synonym (keyword + synonym required)
      add      - Add a manual synonym (keyword + synonym required)
      remove   - Remove all synonyms for keyword(s)

    By default uses local Ollama for generation. Use --openai for OpenAI API.

    Examples:
      yt-rag synonyms list
      yt-rag synonyms generate                    # Generate with local LLM
      yt-rag synonyms generate --openai           # Generate with OpenAI
      yt-rag synonyms remove car truck vehicle
      yt-rag synonyms add -k mpg -s "fuel economy"
    """
    from .config import DEFAULT_SYNONYMS
    from .search import refresh_synonyms_cache

    db = get_db()

    if action == "list":
        if pending:
            syns = db.list_pending_synonyms(limit=50)
            console.print(f"[bold]Pending synonyms ({len(syns)}):[/bold]\n")
            for s in syns:
                console.print(f"  {s.keyword} -> {s.synonym} [dim]({s.source})[/dim]")
        else:
            all_syns = db.get_all_synonyms(approved_only=False)
            if not all_syns:
                # Show defaults
                console.print("[bold]Using default synonyms:[/bold]\n")
                for kw, syns_list in DEFAULT_SYNONYMS.items():
                    console.print(f"  [green]{kw}[/green] -> {', '.join(syns_list)}")
            else:
                console.print(f"[bold]Synonyms ({len(all_syns)} keywords):[/bold]\n")
                for kw, syns_list in all_syns.items():
                    console.print(f"  [green]{kw}[/green] -> {', '.join(syns_list)}")

    elif action == "generate":
        from .keywords import generate_synonyms_with_llm

        # Get top keywords from database or generate fresh
        top_keywords = db.get_keywords(limit=limit)
        if not top_keywords:
            msg = "No keywords in database. Run 'yt-rag keywords --save' first."
            console.print(f"[yellow]{msg}[/yellow]")
            db.close()
            return

        use_local = not openai
        backend_name = "local Ollama" if use_local else "OpenAI"
        msg = f"Generating synonyms for top {len(top_keywords)} keywords using {backend_name}:"
        console.print(f"[bold]{msg}[/bold]\n")

        # Get keywords as list of strings
        keyword_strings = [kw.keyword for kw in top_keywords]

        # Generate synonyms with LLM
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating synonyms with LLM...", total=None)
            llm_synonyms = generate_synonyms_with_llm(
                keyword_strings,
                category="general",
                use_local=use_local,
                model=model,
            )

        count = 0
        for kw_str, synonyms_list in llm_synonyms.items():
            if synonyms_list:
                console.print(f"  [green]{kw_str}[/green] -> {', '.join(synonyms_list)}")
                for syn in synonyms_list:
                    db.add_synonym(kw_str, syn, source="llm", approved=False)
                    count += 1

        console.print(f"\n[green]✓[/green] Added {count} synonym suggestions (pending approval)")
        console.print("Use 'yt-rag synonyms list --pending' to review")

    elif action == "approve":
        if not keyword or not synonym:
            console.print("[red]Error: --keyword and --synonym required for approve[/red]")
            db.close()
            return
        db.approve_synonym(keyword, synonym)
        refresh_synonyms_cache()
        console.print(f"[green]✓[/green] Approved: {keyword} -> {synonym}")

    elif action == "reject":
        if not keyword or not synonym:
            console.print("[red]Error: --keyword and --synonym required for reject[/red]")
            db.close()
            return
        db.reject_synonym(keyword, synonym)
        console.print(f"[green]✓[/green] Rejected: {keyword} -> {synonym}")

    elif action == "add":
        if not keyword or not synonym:
            console.print("[red]Error: --keyword and --synonym required for add[/red]")
            db.close()
            return
        db.add_synonym(keyword, synonym, source="manual", approved=True)
        refresh_synonyms_cache()
        console.print(f"[green]✓[/green] Added: {keyword} -> {synonym}")

    elif action == "remove":
        # Get keywords from positional args or --keyword option
        kws_to_remove = list(keywords) if keywords else []
        if keyword:
            kws_to_remove.append(keyword)

        if not kws_to_remove:
            console.print("[red]Error: specify keywords to remove[/red]")
            console.print("Usage: yt-rag synonyms remove keyword1 keyword2 ...")
            db.close()
            return

        total_removed = 0
        for kw in kws_to_remove:
            count = db.remove_synonyms_for_keyword(kw)
            if count > 0:
                console.print(f"[green]✓[/green] Removed {count} synonyms for '{kw}'")
                total_removed += count
            else:
                console.print(f"[dim]No synonyms found for '{kw}'[/dim]")

        if total_removed > 0:
            refresh_synonyms_cache()
            console.print(f"\n[green]✓[/green] Total: removed {total_removed} synonyms")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: list, generate, approve, reject, add, remove")

    db.close()


@app.command()
def chat(
    video: str = typer.Option(None, "-v", "--video", help="Filter to specific video ID"),
    channel: str = typer.Option(None, "-c", "--channel", help="Filter to specific channel ID"),
    top_k: int = typer.Option(10, "-k", "--top-k", help="Number of sections to retrieve"),
    model: str = typer.Option(None, "-m", "--model", help="Override default model"),
    openai: bool = typer.Option(False, "--openai", help="Use OpenAI API instead of local Ollama"),
    new: bool = typer.Option(False, "--new", help="Start a new chat session"),
    session: str = typer.Option(None, "-s", "--session", help="Resume session by ID prefix"),
    list_sessions: bool = typer.Option(False, "--list", help="List recent chat sessions"),
    history: int = typer.Option(10, "--history", help="Messages to include for context"),
):
    """Interactive chat with your video library.

    By default uses local Ollama. Use --openai to use OpenAI API.
    Use --model to override the default model for either backend.

    Sessions persist conversation history:
      --new         Start a fresh session
      --session ID  Resume a specific session (prefix match)
      --list        Show recent sessions

    Follow-up questions are automatically detected (e.g., "tell me more about the first one").

    Without flags, resumes the most recent session or creates a new one.
    """
    import asyncio
    import readline

    from .config import CHAT_HISTORY_FILE
    from .openai_client import check_ollama_running

    # Load readline history from file
    try:
        readline.read_history_file(CHAT_HISTORY_FILE)
    except FileNotFoundError:
        pass  # First run, no history yet
    readline.set_history_length(1000)
    from .service import ChatSessionManager, RAGService, SearchHit, VideoHit

    db = get_db()

    # Handle --list: show sessions and exit
    if list_sessions:
        session_mgr = ChatSessionManager(db)
        sessions = session_mgr.list_sessions(limit=20)
        if not sessions:
            console.print("[yellow]No chat sessions found[/yellow]")
            db.close()
            return

        table = Table(title="Chat Sessions")
        table.add_column("ID", style="dim")
        table.add_column("Title", max_width=50)
        table.add_column("Messages")
        table.add_column("Updated")

        for s in sessions:
            msg_count = db.get_chat_message_count(s.id)
            updated = s.updated_at.strftime("%m-%d %H:%M") if s.updated_at else ""
            table.add_row(s.id[:8], s.title, str(msg_count), updated)

        console.print(table)
        db.close()
        return

    # Determine backend and model
    use_local = not openai
    if model:
        chat_model = model
    else:
        chat_model = DEFAULT_OLLAMA_MODEL if use_local else DEFAULT_CHAT_MODEL

    # Check Ollama is running if using local backend
    if use_local and not check_ollama_running():
        console.print("[red]Error: Ollama is not running[/red]")
        console.print("Start it with: sudo systemctl start ollama")
        console.print("Or use --openai to use OpenAI API")
        db.close()
        raise typer.Exit(1)

    service = RAGService(use_local=use_local)
    session_mgr = ChatSessionManager(db)

    # Determine session to use
    if new:
        # Explicitly create new session
        current_session = session_mgr.create_session(video_id=video, channel_id=channel)
        console.print(f"[dim]New session: {current_session.id[:8]}[/dim]")
    elif session:
        # Resume specific session
        current_session = session_mgr.load_session(session)
        if not current_session:
            console.print(f"[red]Session not found: {session}[/red]")
            db.close()
            raise typer.Exit(1)
        msg_count = session_mgr.get_message_count()
        console.print(f"[dim]Resumed: {current_session.title} ({msg_count} messages)[/dim]")
    else:
        # Auto: try most recent, or create new
        current_session = session_mgr.get_most_recent_session()
        if current_session:
            msg_count = session_mgr.get_message_count()
            console.print(f"[dim]Resumed: {current_session.title} ({msg_count} messages)[/dim]")
        else:
            current_session = session_mgr.create_session(video_id=video, channel_id=channel)
            console.print(f"[dim]New session: {current_session.id[:8]}[/dim]")

    # Show model info
    backend_name = "Ollama" if use_local else "OpenAI"
    console.print(f"[bold]yt-rag Chat[/bold] [dim]({backend_name})[/dim]")
    console.print(f"[dim]Model: {chat_model}[/dim]")

    console.print("Type your questions. Use 'exit' or Ctrl+C to quit.")
    console.print("[dim]Commands: /new, /sessions, /rename <title>[/dim]\n")

    # Cache for search results to support follow-up questions
    cached_search_hits: list[SearchHit] = []

    async def run_chat():
        nonlocal current_session, cached_search_hits

        while True:
            try:
                # Use input() with prompt to prevent deletion
                query = input("> ")
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye!")
                break

            query = query.strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                console.print("Goodbye!")
                break

            # Handle in-chat commands
            if query.startswith("/"):
                cmd_parts = query[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd == "new":
                    current_session = session_mgr.create_session(video_id=video, channel_id=channel)
                    console.print(f"[green]Started new session: {current_session.id[:8]}[/green]\n")
                elif cmd == "sessions":
                    sessions = session_mgr.list_sessions(limit=10)
                    if not sessions:
                        console.print("[yellow]No sessions[/yellow]\n")
                    else:
                        for s in sessions:
                            marker = "*" if s.id == current_session.id else " "
                            msg_count = db.get_chat_message_count(s.id)
                            console.print(f"{marker} {s.id[:8]}: {s.title} ({msg_count} msgs)")
                        console.print()
                elif cmd == "rename" and cmd_arg:
                    session_mgr.rename_session(cmd_arg)
                    console.print(f"[green]Renamed to: {cmd_arg}[/green]\n")
                elif cmd == "rename":
                    console.print("[yellow]Usage: /rename <new title>[/yellow]\n")
                else:
                    console.print(f"[yellow]Unknown command: /{cmd}[/yellow]\n")
                continue

            # Save user message
            session_mgr.add_user_message(query)

            # Parse "top N" from query to override top_k
            import re

            query_top_k = top_k
            top_n_match = re.search(r"\btop\s+(\d+)\b", query.lower())
            if top_n_match:
                query_top_k = int(top_n_match.group(1))

            # Get conversation history for context
            conv_history = session_mgr.get_messages_for_llm(limit=history)
            # Remove the last message (current query) since we pass it separately
            if conv_history:
                conv_history = conv_history[:-1]

            # Stream the response
            console.print()
            answer_started = False
            answer_chunks: list[str] = []

            section_hits: list[SearchHit] = []
            video_summary_count = 0
            latency_ms = 0
            async for item in service.ask_stream(
                query=query,
                top_k=query_top_k,
                video_id=video or current_session.video_id,
                channel_id=channel or current_session.channel_id,
                chat_model=chat_model,
                use_ollama=not openai,
                ollama_model=chat_model,
                conversation_history=conv_history if conv_history else None,
                cached_hits=cached_search_hits if cached_search_hits else None,
            ):
                if isinstance(item, VideoHit):
                    # Track video summaries from phase 1
                    video_summary_count += 1
                elif isinstance(item, SearchHit):
                    # Collect section hits for display
                    section_hits.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "search_done":
                        count = item.get("count", 0)
                        summaries = item.get("video_summaries", 0)
                        if count == 0 and summaries == 0:
                            console.print("[yellow]No relevant sources found.[/yellow]")
                    elif item.get("type") == "followup":
                        # Follow-up query detected - results will come from cached hits
                        console.print("[dim](Using previous search results)[/dim]")
                    elif item.get("type") == "error":
                        console.print(f"[red]{item.get('message')}[/red]")
                    elif item.get("type") == "done":
                        latency_ms = item.get("latency_ms", 0)
                        # Cache section hits for follow-up questions
                        new_hits = item.get("section_hits", [])
                        if new_hits:
                            cached_search_hits = new_hits
                elif isinstance(item, str):
                    # Stream answer chunks
                    if not answer_started:
                        answer_started = True
                    console.print(item, end="")
                    answer_chunks.append(item)

            if answer_started:
                console.print()  # Newline after streaming
                # Save assistant response
                full_answer = "".join(answer_chunks)
                session_mgr.add_assistant_message(full_answer)

            # Show results list with rich formatting (no "Sources" header)
            if section_hits:
                # Count unique videos in results
                unique_videos = len({hit.video_id for hit in section_hits})
                s_suffix = "s" if unique_videos != 1 else ""
                sec_count = len(section_hits)
                found_msg = f"Found {unique_videos} relevant video{s_suffix} ({sec_count} sections)"
                console.print(f"\n[dim]{found_msg}[/dim]")
                console.print()
                for i, hit in enumerate(section_hits, 1):
                    if hit.channel_name:
                        channel_str = f"[magenta]{hit.channel_name}[/magenta] | "
                    else:
                        channel_str = ""
                    if hit.published_at:
                        date_str = f" [dim]({hit.published_at.strftime('%Y-%m-%d')})[/dim]"
                    else:
                        date_str = ""
                    content_preview = hit.section.content[:150].replace("\n", " ").strip()
                    if len(hit.section.content) > 150:
                        content_preview += "..."

                    title_line = (
                        f"[cyan]{i}.[/cyan] {channel_str}[bold]{hit.video_title}[/bold]{date_str}"
                    )
                    console.print(title_line)
                    console.print(f"   Section: {hit.section.title}")
                    console.print(f"   [dim]{content_preview}[/dim]")
                    console.print(f"   [link={hit.timestamp_url}]{hit.timestamp_url}[/link]")
                    console.print()

            console.print(f"\n[dim]({latency_ms}ms)[/dim]\n")

    try:
        asyncio.run(run_chat())
    finally:
        readline.write_history_file(CHAT_HISTORY_FILE)
        service.close()
        db.close()


@app.command("test-generate")
def test_generate(
    step: str = typer.Option(
        "all",
        "--step",
        "-s",
        help="Step to run: prepare, analyze, build, or all",
    ),
    videos_per_channel: int = typer.Option(
        5,
        "--videos",
        "-n",
        help="Videos to sample per channel (prepare step)",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        "-l",
        help="Max videos to analyze (analyze step)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model for analysis (default: qwen2.5:7b-instruct/gpt-4o-mini)",
    ),
    openai: bool = typer.Option(
        False,
        "--openai",
        help="Use OpenAI API instead of local Ollama",
    ),
):
    """Generate benchmark test cases from video transcripts.

    Three-step workflow:
    1. prepare: Sample videos, save raw data to tests/data/raw_videos.json
    2. analyze: LLM extracts entities/topics/facts to tests/data/video_analysis_{model}.json
    3. build: Generate test queries to tests/data/benchmark_generated_{model}.json

    Supports both local (Ollama) and remote (OpenAI) LLMs. For OpenAI, sections are
    analyzed individually to reduce cost and context usage.

    Examples:
        yt-rag test-generate                    # Run all steps with local LLM
        yt-rag test-generate --step=prepare    # Only sample videos
        yt-rag test-generate --step=analyze --limit=10  # Analyze 10 videos
        yt-rag test-generate --openai          # Use OpenAI API (gpt-4o-mini)
        yt-rag test-generate --openai --model=gpt-4o  # Use specific OpenAI model
        yt-rag test-generate --model=qwen2.5:7b-instruct  # Use specific local model
    """
    from .config import DEFAULT_CHAT_MODEL, DEFAULT_OLLAMA_MODEL
    from .test_generate import analyze_videos, build_tests, prepare_raw_data

    db = get_db()

    # Determine model to use
    if model is None:
        model = DEFAULT_CHAT_MODEL if openai else DEFAULT_OLLAMA_MODEL

    if step in ("all", "prepare"):
        console.print("[bold]Step 1: Preparing raw data...[/bold]")
        result = prepare_raw_data(db, videos_per_channel=videos_per_channel)
        console.print(
            f"  Sampled {result.videos_sampled} videos from {result.channels_sampled} channels"
        )
        console.print(f"  Output: {result.output_file}")

    if step in ("all", "analyze"):
        api_type = "OpenAI" if openai else "Ollama"
        console.print(f"\n[bold]Step 2: Analyzing videos with {api_type} ({model})...[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Analyzing...", total=None)
            result = analyze_videos(limit=limit, model=model, use_openai=openai)
        console.print(f"  Analyzed {result.videos_analyzed} videos")
        extracted = (
            f"{result.total_entities} entities, "
            f"{result.total_topics} topics, "
            f"{result.total_facts} facts"
        )
        console.print(f"  Extracted: {extracted}")
        console.print(f"  Output: {result.output_file}")

    if step in ("all", "build"):
        console.print("\n[bold]Step 3: Building test cases...[/bold]")
        result = build_tests(model=model)
        console.print(f"  Generated {result.tests_generated} test cases")
        console.print(f"  By type: {result.by_type}")
        console.print(f"  Output: {result.output_file}")

    db.close()
    console.print("\n[green]Done![/green]")


@app.command("test-report")
def test_report(
    results_file: str = typer.Option(
        "tests/data/benchmark_results_gpt-4o.json",
        "--results",
        "-r",
        help="Path to benchmark results JSON file",
    ),
    tests_file: str = typer.Option(
        None, "--tests", "-t", help="Path to benchmark tests JSON file (auto-detected from results)"
    ),
    output: str = typer.Option(
        "tests/data/benchmark_report.html", "--output", "-o", help="Output HTML file path"
    ),
    filter_status: str = typer.Option(
        None, "--filter", help="Filter results: 'pass', 'fail', 'disagree', 'empty', 'meta'"
    ),
):
    """Generate HTML report for benchmark results.

    Creates a detailed HTML report showing:
    - Test origin (channel, video, search type, keywords)
    - RAG answer
    - Validation results from both Qwen and GPT-4o
    - Keyword matching results

    Examples:
        yt-rag test-report                          # Default files
        yt-rag test-report --filter=disagree        # Only show disagreements
        yt-rag test-report --filter=fail            # Only show failures
        yt-rag test-report -o report.html           # Custom output path
    """
    import json
    from pathlib import Path

    results_path = Path(results_file)
    if not results_path.exists():
        console.print(f"[red]Results file not found: {results_path}[/red]")
        raise typer.Exit(1)

    with open(results_path) as f:
        results_data = json.load(f)

    # Auto-detect tests file from results source_file
    if tests_file is None:
        source = results_data.get("source_file", "")
        if source:
            tests_path = Path(source)
        else:
            # Guess based on results filename
            tests_path = results_path.parent / results_path.name.replace("results", "generated")
    else:
        tests_path = Path(tests_file)

    if not tests_path.exists():
        console.print(f"[red]Tests file not found: {tests_path}[/red]")
        console.print("Use --tests to specify the benchmark tests file")
        raise typer.Exit(1)

    with open(tests_path) as f:
        tests_data = json.load(f)

    # Build lookup from tests
    tests_lookup = {}
    for test in tests_data.get("test_cases", []):
        tests_lookup[test["query"]] = test

    # Merge results with test metadata
    merged = []
    for result in results_data.get("results", []):
        query = result["query"]
        test_meta = tests_lookup.get(query, {})
        merged.append(
            {
                **result,
                "channel": test_meta.get("channel", "unknown"),
                "source_video": test_meta.get("source_video", "unknown"),
                "note": test_meta.get("note", ""),
                "expected_video_ids": test_meta.get("expected_video_ids", []),
            }
        )

    # Apply filter
    if filter_status:
        if filter_status == "disagree":
            merged = [
                m
                for m in merged
                if m.get("validation_qwen", {}).get("pass")
                != m.get("validation_gpt4o", {}).get("pass")
            ]
        elif filter_status == "pass":
            merged = [
                m
                for m in merged
                if m.get("validation_gpt4o", {}).get("pass")
                and m.get("validation_qwen", {}).get("pass")
            ]
        elif filter_status == "fail":
            merged = [
                m
                for m in merged
                if not m.get("validation_gpt4o", {}).get("pass")
                or not m.get("validation_qwen", {}).get("pass")
            ]
        elif filter_status == "empty":
            merged = [m for m in merged if not m.get("answer")]
        elif filter_status == "meta":
            merged = [m for m in merged if m.get("expected_type") == "meta"]

    # Generate HTML
    summary = results_data.get("summary", {})
    html = _generate_html_report(merged, summary, results_data, filter_status)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    console.print(f"[green]Report generated: {output_path}[/green]")
    console.print(f"  Total results: {len(merged)}")
    if filter_status:
        console.print(f"  Filter: {filter_status}")


def _generate_html_report(
    results: list, summary: dict, meta: dict, filter_status: str | None
) -> str:
    """Generate HTML report content."""
    import html

    def esc(s):
        return html.escape(str(s)) if s else ""

    def status_badge(passed: bool | None, reason: str = "") -> str:
        if passed is None:
            return '<span class="badge badge-unknown">?</span>'
        elif passed:
            return f'<span class="badge badge-pass" title="{esc(reason)}">✓ Pass</span>'
        else:
            return f'<span class="badge badge-fail" title="{esc(reason)}">✗ Fail</span>'

    def agree_badge(qwen_pass: bool | None, gpt_pass: bool | None) -> str:
        if qwen_pass == gpt_pass:
            return '<span class="badge badge-agree">Agree</span>'
        elif gpt_pass and not qwen_pass:
            return '<span class="badge badge-disagree">GPT✓ Qwen✗</span>'
        else:
            return '<span class="badge badge-disagree">Qwen✓ GPT✗</span>'

    # Summary stats
    llm_val = summary.get("llm_validation", {})
    qwen_stats = llm_val.get("qwen", {})
    gpt_stats = llm_val.get("gpt4o", {})
    kw_stats = summary.get("keyword_match", {})
    class_stats = summary.get("classification", {})

    filter_desc = f" (filtered: {filter_status})" if filter_status else ""

    # Pre-compute filter info HTML (avoid backslash in f-string for Python 3.11)
    if filter_status:
        filter_info_html = (
            "<div class='filter-info'>Filtered: "
            + esc(filter_status)
            + f" ({len(results)} results)</div>"
        )
    else:
        filter_info_html = ""

    rows_html = []
    for i, r in enumerate(results, 1):
        qwen_val = r.get("validation_qwen", {})
        gpt_val = r.get("validation_gpt4o", {})

        answer = r.get("answer", "")
        if not answer:
            answer_html = '<span class="empty-answer">(empty)</span>'
        else:
            answer_html = f'<div class="answer">{esc(answer)}</div>'

        keywords_found = r.get("keywords_found", [])
        keywords_missing = r.get("keywords_missing", [])
        kw_html = ""
        if keywords_found:
            kw_html += " ".join(f'<span class="kw-found">{esc(k)}</span>' for k in keywords_found)
        if keywords_missing:
            kw_html += " ".join(
                f'<span class="kw-missing">{esc(k)}</span>' for k in keywords_missing
            )

        # Handle None values for META queries
        channel_str = r.get("channel") or "(global)"
        video_str = r.get("source_video") or ""

        row = f"""
        <tr class="result-row">
            <td class="num">{i}</td>
            <td class="meta">
                <div class="query"><strong>{esc(r.get("query", ""))}</strong></div>
                <div class="channel">📺 {esc(channel_str)}</div>
                {"<div class='video'>🎬 " + esc(video_str) + "</div>" if video_str else ""}
                <div class="type">🏷️ {esc(r.get("expected_type", ""))}</div>
                <div class="keywords">🔑 {kw_html if kw_html else "(none)"}</div>
            </td>
            <td class="answer-cell">{answer_html}</td>
            <td class="validation">
                <div class="val-qwen">
                    <strong>Qwen:</strong> \
{status_badge(qwen_val.get("pass"), qwen_val.get("reason", ""))}
                    <div class="reason">{esc(qwen_val.get("reason", ""))}</div>
                </div>
                <div class="val-gpt">
                    <strong>GPT-4o:</strong> \
{status_badge(gpt_val.get("pass"), gpt_val.get("reason", ""))}
                    <div class="reason">{esc(gpt_val.get("reason", ""))}</div>
                </div>
                <div class="agreement">
                    {agree_badge(qwen_val.get("pass"), gpt_val.get("pass"))}
                </div>
            </td>
            <td class="type-check">
                {"✓" if r.get("type_passed") else "✗"} {esc(r.get("got_type", ""))}
            </td>
        </tr>
        """
        rows_html.append(row)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report{filter_desc}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            font-size: 14px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-box .value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-box .label {{ color: #666; font-size: 12px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background: #333;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 500;
        }}
        td {{
            padding: 12px 8px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }}
        tr:hover {{ background: #f8f9fa; }}
        .num {{ width: 40px; text-align: center; color: #999; }}
        .meta {{ width: 280px; }}
        .query {{ font-size: 15px; margin-bottom: 8px; }}
        .channel, .video, .type, .keywords {{ font-size: 12px; color: #666; margin: 2px 0; }}
        .video {{
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 250px;
        }}
        .answer-cell {{ width: 35%; }}
        .answer {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-size: 13px;
            line-height: 1.5;
            max-height: 200px;
            overflow-y: auto;
        }}
        .empty-answer {{ color: #999; font-style: italic; }}
        .validation {{ width: 300px; }}
        .val-qwen, .val-gpt {{ margin-bottom: 10px; }}
        .reason {{ font-size: 11px; color: #666; margin-top: 4px; }}
        .agreement {{ margin-top: 8px; }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }}
        .badge-pass {{ background: #d4edda; color: #155724; }}
        .badge-fail {{ background: #f8d7da; color: #721c24; }}
        .badge-agree {{ background: #d1ecf1; color: #0c5460; }}
        .badge-disagree {{ background: #fff3cd; color: #856404; }}
        .badge-unknown {{ background: #e2e3e5; color: #383d41; }}
        .type-check {{ width: 80px; text-align: center; }}
        .kw-found {{
            background: #d4edda;
            color: #155724;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            margin: 1px;
            display: inline-block;
        }}
        .kw-missing {{
            background: #f8d7da;
            color: #721c24;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            margin: 1px;
            display: inline-block;
        }}
        .filter-info {{
            background: #fff3cd;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 RAG Benchmark Report</h1>
        <p>Pipeline: {esc(meta.get("pipeline", ""))} | \
Validators: {", ".join(meta.get("validators", []))}</p>

        <div class="summary">
            <div class="summary-grid">
                <div class="stat-box">
                    <div class="value">{meta.get("total_tests", 0)}</div>
                    <div class="label">Total Tests</div>
                </div>
                <div class="stat-box">
                    <div class="value">{class_stats.get("accuracy", 0):.1f}%</div>
                    <div class="label">Classification \
({class_stats.get("correct", 0)}/{class_stats.get("total", 0)})</div>
                </div>
                <div class="stat-box">
                    <div class="value">{kw_stats.get("accuracy", 0):.1f}%</div>
                    <div class="label">Keyword Match \
({kw_stats.get("hits", 0)}/{kw_stats.get("total", 0)})</div>
                </div>
                <div class="stat-box">
                    <div class="value">{qwen_stats.get("rate", 0):.1f}%</div>
                    <div class="label">Qwen Pass ({qwen_stats.get("pass", 0)})</div>
                </div>
                <div class="stat-box">
                    <div class="value">{gpt_stats.get("rate", 0):.1f}%</div>
                    <div class="label">GPT-4o Pass ({gpt_stats.get("pass", 0)})</div>
                </div>
            </div>
        </div>

        {filter_info_html}

        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Test Info</th>
                    <th>RAG Answer</th>
                    <th>Validation</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    app()
