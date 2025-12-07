"""CLI commands for yt-rag."""

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ensure_data_dir,
    get_yt_dlp_delay,
)
from .db import Database
from .discovery import extract_video_id, get_channel_info, get_video_info, list_channel_videos
from .embed import embed_all_sections, embed_all_summaries, embed_video, get_index_stats
from .eval import add_feedback, add_test_case, run_benchmark
from .export import export_all_chunks, export_to_json, export_to_jsonl
from .search import search as rag_search
from .summarize import summarize_video
from .transcript import TranscriptUnavailable, fetch_transcript

app = typer.Typer(
    name="yt-rag",
    help="Extract YouTube transcripts for RAG pipelines.",
    no_args_is_help=True,
)
console = Console()


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
    force_fetch: bool = typer.Option(
        False, "--force-fetch", help="Re-fetch all transcripts, not just pending"
    ),
    force_refresh_meta: bool = typer.Option(
        False, "--force-refresh-meta", help="Force refresh all metadata, ignoring timestamps"
    ),
    skip_sync: bool = typer.Option(False, "--skip-sync", help="Skip syncing channels"),
    skip_embed: bool = typer.Option(False, "--skip-embed", help="Skip embedding step"),
    workers: int = typer.Option(
        DEFAULT_FETCH_WORKERS, "-w", "--workers", help="Parallel workers for transcript fetch"
    ),
    openai: bool = typer.Option(False, "--openai", help="Use OpenAI for embeddings"),
):
    """Run the full update pipeline.

    This command runs the complete pipeline to update your library:
    1. sync-channel: Pull new videos from tracked channels
    2. fetch-transcript: Fetch transcripts for pending videos
    3. refresh-meta: Refresh metadata (skips if refreshed within 1 day)
    4. process-transcript: Sectionize and summarize videos
    5. embed: Build/update vector index

    Examples:
        yt-rag update                      # Run full pipeline
        yt-rag update --skip-sync          # Skip channel sync
        yt-rag update --force-fetch        # Re-fetch ALL transcripts
        yt-rag update --force-refresh-meta # Force refresh all metadata
        yt-rag update --openai             # Use OpenAI for embeddings
    """
    from .openai_client import check_ollama_running

    use_local = not openai

    # Check Ollama is running if using local embeddings
    if use_local and not check_ollama_running():
        console.print("[red]Error: Ollama is not running[/red]")
        console.print("Start it with: sudo systemctl start ollama")
        console.print("Or use --openai to use OpenAI embeddings")
        raise typer.Exit(1)

    db = get_db()

    # Step 1: Sync channels
    if not skip_sync:
        console.print("\n[bold]Step 1: Syncing channels[/bold]")
        channels = db.list_channels()
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

    # Step 2: Fetch transcripts (uses youtube_transcript_api - no rate limiting needed)
    console.print("\n[bold]Step 2: Fetching transcripts[/bold]")
    if force_fetch:
        # Re-fetch all videos with fetched status
        videos_to_fetch = db.list_videos(status="fetched")
        # Also include pending
        videos_to_fetch.extend(db.get_pending_videos())
        console.print(f"[yellow]Force mode: re-fetching {len(videos_to_fetch)} videos[/yellow]")
    else:
        videos_to_fetch = db.get_pending_videos()

    if not videos_to_fetch:
        console.print("[green]✓[/green] No pending videos to fetch")
    else:
        fetched = 0
        unavailable = 0
        errors = 0
        lock = threading.Lock()

        def process_video_fetch(video):
            """Fetch transcript only - metadata is handled separately in Step 3."""
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

    # Step 3: Refresh metadata (uses yt-dlp - sequential with rate limiting)
    console.print("\n[bold]Step 3: Refreshing metadata[/bold]")
    import time
    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(days=METADATA_FRESHNESS_DAYS)

    # Channels missing metadata
    channels = [c for c in db.list_channels() if not c.description]
    channel_errors = 0
    if channels:
        for channel in channels:
            time.sleep(get_yt_dlp_delay())  # Random rate limiting
            try:
                updated_channel = get_channel_info(channel.url)
                db.add_channel(updated_channel)
                console.print(f"[green]✓[/green] {channel.name}")
            except Exception as e:
                channel_errors += 1
                console.print(f"[red]Error[/red] {channel.name}: {e}")
        if channel_errors > 0:
            console.print(f"[yellow]Channel errors: {channel_errors}[/yellow]")
    else:
        console.print("[green]✓[/green] All channels have metadata")

    # Videos needing metadata refresh:
    # - No metadata_refreshed_at (NULL) = never refreshed
    # - metadata_refreshed_at older than 1 day
    # - force_refresh_meta = refresh all
    all_fetched = db.list_videos(status="fetched")
    if force_refresh_meta:
        videos_to_refresh = all_fetched
        console.print(
            f"[yellow]Force mode: refreshing all {len(videos_to_refresh)} videos[/yellow]"
        )
    else:
        videos_to_refresh = [
            v for v in all_fetched
            if v.metadata_refreshed_at is None or v.metadata_refreshed_at < cutoff
        ]

    if videos_to_refresh:
        updated = 0
        errors = 0
        skipped = len(all_fetched) - len(videos_to_refresh)

        if skipped > 0:
            console.print(
                f"[dim]Skipping {skipped} videos refreshed within "
                f"{METADATA_FRESHNESS_DAYS} day(s)[/dim]"
            )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Refreshing video metadata", total=len(videos_to_refresh))

            for video in videos_to_refresh:
                time.sleep(get_yt_dlp_delay())  # Random rate limiting
                title = video.title[:40] + "..." if len(video.title) > 40 else video.title

                try:
                    result = get_video_info(video.url)
                    db.add_video(result)
                    updated += 1
                    progress.update(task, description=f"[green]✓[/green] {title}")
                except Exception as e:
                    errors += 1
                    console.print(f"\n[red]Error[/red] {title}: {e}")

                progress.advance(task)

        console.print(f"[green]✓[/green] Updated {updated} videos, {errors} errors")
    else:
        console.print(
            f"[green]✓[/green] All videos refreshed within {METADATA_FRESHNESS_DAYS} day(s)"
        )

    # Step 4: Process transcripts (sectionize + summarize)
    console.print("\n[bold]Step 4: Processing transcripts[/bold]")
    videos = db.list_videos(status="fetched")

    # Filter to videos needing processing
    videos_to_process = []
    for v in videos:
        existing_sections = db.get_sections(v.id)
        existing_summary = db.get_summary(v.id)
        if not existing_sections or not existing_summary:
            videos_to_process.append(v)

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
                        sectionize_video(video.id, db)
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
                        summarize_video(video.id, db)
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

        stats = db.get_stats()
        if stats["sections"] == 0:
            console.print("[yellow]No sections to embed[/yellow]")
        else:
            # Embed sections
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Embedding sections...", total=None)
                result = embed_all_sections(db, model=None, rebuild=False, use_local=use_local)

            if result.sections_embedded > 0:
                console.print(
                    f"[green]✓[/green] Embedded {result.sections_embedded} sections "
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
                        db, model=None, rebuild=False, use_local=use_local
                    )

                if summary_result.sections_embedded > 0:
                    console.print(
                        f"[green]✓[/green] Embedded {summary_result.sections_embedded} summaries "
                        f"({summary_result.tokens_used} tokens)"
                    )
                else:
                    console.print("[green]✓[/green] All summaries already embedded")
    else:
        console.print("\n[bold]Step 5: Building embeddings[/bold] [dim](skipped)[/dim]")

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
    gpt_titles: bool = typer.Option(False, "--gpt-titles", help="Generate GPT titles for chunks"),
    model: str = typer.Option(DEFAULT_CHAT_MODEL, "-m", "--model", help="Model for GPT operations"),
    force: bool = typer.Option(False, "--force", help="Re-process even if already done"),
):
    """Process videos: sectionize (using YouTube chapters) and summarize.

    Sectionizing uses YouTube chapters when available, falling back to
    time-based chunks. Use --gpt-titles to generate descriptive titles for
    videos without chapters.
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
                            video.id, db, generate_titles=gpt_titles, model=model
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
                        summarize_video(video.id, db, model)
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

    By default uses local Ollama embeddings (nomic-embed-text).
    Use --openai to use OpenAI embeddings (text-embedding-3-small).

    Local and OpenAI indexes are stored separately, so you can switch between them.
    """
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

    console.print(f"[dim]Using {backend_name} embeddings[/dim]")

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
            f"[green]✓[/green] Embedded {result.sections_embedded} sections "
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

        total_tokens = 0

        # Embed sections
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if rebuild:
                progress.add_task("Rebuilding sections index...", total=None)
            else:
                progress.add_task("Embedding new sections...", total=None)

            result = embed_all_sections(db, model, rebuild=rebuild, use_local=use_local)
            total_tokens += result.tokens_used

        if result.sections_embedded > 0:
            console.print(
                f"[green]✓[/green] Embedded {result.sections_embedded} sections "
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

            if summary_result.sections_embedded > 0:
                console.print(
                    f"[green]✓[/green] Embedded {summary_result.sections_embedded} video summaries "
                    f"({summary_result.tokens_used} tokens)"
                )
            else:
                console.print("[green]✓[/green] All summaries already embedded")

    # Show index stats
    idx_stats = get_index_stats(use_local=use_local)
    sec_count = idx_stats['sections_vectors']
    sum_count = idx_stats['summaries_vectors']
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
    model: str = typer.Option(DEFAULT_CHAT_MODEL, "-m", "--model", help="Chat model"),
    fast: bool = typer.Option(False, "--fast", help="Use gpt-3.5-turbo for faster response"),
    openai: bool = typer.Option(False, "--openai", help="Use OpenAI for embeddings"),
):
    """Ask a question about video content using RAG."""
    chat_model = "gpt-3.5-turbo" if fast else model
    use_local = not openai
    db = get_db()

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
            generate_answer=not no_answer,
            chat_model=chat_model,
            use_local=use_local,
        )

    if not result.hits:
        console.print("[yellow]No relevant content found.[/yellow]")
        db.close()
        return

    # Show answer if generated
    if result.answer:
        console.print("\n[bold]Answer:[/bold]")
        console.print(result.answer)
        console.print()

    # Show sources
    console.print(f"[bold]Sources ({len(result.hits)}):[/bold]")
    for i, hit in enumerate(result.hits, 1):
        score_pct = int(hit.score * 100)
        console.print(f"\n[cyan]{i}.[/cyan] {hit.video_title}")
        console.print(f"   Section: {hit.section.title}")
        console.print(f"   Score: {score_pct}% | [link={hit.timestamp_url}]{hit.timestamp_url}[/]")

    # Show stats
    model_info = f" | Model: {chat_model}" if not no_answer else ""
    console.print(
        f"\n[dim]Latency: {result.latency_ms}ms | "
        f"Tokens: {result.tokens_embedding} embed + {result.tokens_chat} chat{model_info}[/dim]"
    )

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

    Rate limiting is controlled by YT_DLP_DELAY in .env (default: 0.2s).

    Examples:
        yt-rag refresh-meta              # Refresh all (videos + channels)
        yt-rag refresh-meta --video      # Refresh only video metadata
        yt-rag refresh-meta --channel    # Refresh only channel metadata
        yt-rag refresh-meta --force      # Refresh all, even if metadata exists
    """
    import time

    db = get_db()

    # Determine what to refresh
    do_videos = not channel_only
    do_channels = not video_only

    total_updated = 0
    total_errors = 0

    # Refresh channels
    if do_channels:
        channels = db.list_channels()
        if not force:
            channels = [c for c in channels if not c.description]

        if not channels:
            console.print("[green]✓[/green] All channels have metadata")
        else:
            updated = 0
            errors = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Refreshing channel metadata", total=None)

                for channel in channels:
                    time.sleep(get_yt_dlp_delay())  # Random rate limiting
                    try:
                        updated_channel = get_channel_info(channel.url)
                        db.add_channel(updated_channel)
                        updated += 1
                        console.print(f"[green]✓[/green] {channel.name}")
                        if updated_channel.subscriber_count:
                            console.print(f"   Subscribers: {updated_channel.subscriber_count:,}")
                        if updated_channel.tags:
                            console.print(f"   Tags: {', '.join(updated_channel.tags[:5])}")
                    except Exception as e:
                        errors += 1
                        console.print(f"[red]Error[/red] {channel.name}: {e}")

            console.print(f"[green]✓[/green] Updated {updated} channels, {errors} errors")
            total_updated += updated
            total_errors += errors

    # Refresh videos (sequential to respect rate limiting)
    if do_videos:
        videos = db.list_videos(status="fetched")
        if not force:
            videos = [v for v in videos if not v.description]

        if limit:
            videos = videos[:limit]

        if not videos:
            console.print("[green]✓[/green] All videos have metadata")
        else:
            updated = 0
            errors = 0

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Refreshing video metadata", total=len(videos))

                for video in videos:
                    time.sleep(get_yt_dlp_delay())  # Random rate limiting
                    title = video.title[:40] + "..." if len(video.title) > 40 else video.title

                    try:
                        result = get_video_info(video.url)
                        db.add_video(result)
                        updated += 1
                        progress.update(task, description=f"[green]✓[/green] {title}")
                    except Exception as e:
                        errors += 1
                        console.print(f"\n[red]Error[/red] {title}: {e}")

                    progress.advance(task)

            console.print(f"[green]✓[/green] Updated {updated} videos, {errors} errors")
            total_updated += updated
            total_errors += errors

    db.close()


@app.command()
def version():
    """Show version."""
    console.print(f"yt-rag {__version__}")


@app.command()
def chat(
    video: str = typer.Option(None, "-v", "--video", help="Filter to specific video ID"),
    channel: str = typer.Option(None, "-c", "--channel", help="Filter to specific channel ID"),
    top_k: int = typer.Option(10, "-k", "--top-k", help="Number of sections to retrieve"),
    model: str = typer.Option(DEFAULT_OLLAMA_MODEL, "-m", "--model", help="Model name"),
    openai: bool = typer.Option(False, "--openai", help="Use OpenAI instead of local Ollama"),
    new: bool = typer.Option(False, "--new", help="Start a new chat session"),
    session: str = typer.Option(None, "-s", "--session", help="Resume session by ID prefix"),
    list_sessions: bool = typer.Option(False, "--list", help="List recent chat sessions"),
    history: int = typer.Option(10, "--history", help="Messages to include for context"),
):
    """Interactive chat with your video library.

    By default uses local Ollama. Use --openai to use OpenAI API.

    Sessions persist conversation history:
      --new         Start a fresh session
      --session ID  Resume a specific session (prefix match)
      --list        Show recent sessions

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

    # Check Ollama is running unless using OpenAI
    if not openai and not check_ollama_running():
        console.print("[red]Error: Ollama is not running[/red]")
        console.print("Start it with: sudo systemctl start ollama")
        console.print("Or use --openai to use OpenAI API")
        db.close()
        raise typer.Exit(1)

    use_local = not openai
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

    # Use appropriate model based on backend
    if openai:
        chat_model = DEFAULT_CHAT_MODEL if model == DEFAULT_OLLAMA_MODEL else model
        console.print("[bold]yt-rag Chat[/bold] [dim](OpenAI)[/dim]")
        console.print(f"[dim]Model: {chat_model}[/dim]")
    else:
        chat_model = model
        console.print("[bold]yt-rag Chat[/bold] [dim](Ollama)[/dim]")
        console.print(f"[dim]Model: {chat_model}[/dim]")

    console.print("Type your questions. Use 'exit' or Ctrl+C to quit.")
    console.print("[dim]Commands: /new, /sessions, /rename <title>[/dim]\n")

    async def run_chat():
        nonlocal current_session

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

            # Get conversation history for context
            conv_history = session_mgr.get_messages_for_llm(limit=history)
            # Remove the last message (current query) since we pass it separately
            if conv_history:
                conv_history = conv_history[:-1]

            # Stream the response
            console.print()
            answer_started = False
            answer_chunks: list[str] = []

            seen_videos: set[str] = set()
            video_summary_count = 0
            async for item in service.ask_stream(
                query=query,
                top_k=top_k,
                video_id=video or current_session.video_id,
                channel_id=channel or current_session.channel_id,
                chat_model=chat_model,
                use_ollama=not openai,
                ollama_model=chat_model,
                conversation_history=conv_history if conv_history else None,
            ):
                if isinstance(item, VideoHit):
                    # Track video summaries from phase 1
                    video_summary_count += 1
                elif isinstance(item, SearchHit):
                    # Track unique videos from phase 2
                    seen_videos.add(item.video_id)
                elif isinstance(item, dict):
                    if item.get("type") == "search_done":
                        count = item.get("count", 0)
                        summaries = item.get("video_summaries", 0)
                        total_summary_matches = item.get("total_summary_matches", 0)
                        if count == 0 and summaries == 0:
                            console.print("[yellow]No relevant sources found.[/yellow]")
                        else:
                            # Show accurate match count from summaries
                            console.print(
                                f"[dim]Found {total_summary_matches} relevant videos[/dim]\n"
                            )
                    elif item.get("type") == "error":
                        console.print(f"[red]{item.get('message')}[/red]")
                    elif item.get("type") == "done":
                        latency = item.get("latency_ms", 0)
                        console.print(f"\n[dim]({latency}ms)[/dim]\n")
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

    try:
        asyncio.run(run_chat())
    finally:
        readline.write_history_file(CHAT_HISTORY_FILE)
        service.close()
        db.close()


if __name__ == "__main__":
    app()
