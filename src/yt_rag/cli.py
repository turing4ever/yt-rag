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
from .config import (
    DB_PATH,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FETCH_WORKERS,
    ensure_data_dir,
)
from .db import Database
from .discovery import extract_video_id, get_channel_info, get_video_info, list_channel_videos
from .embed import embed_all_sections, embed_video, get_index_stats
from .export import export_all_chunks, export_to_json, export_to_jsonl
from .search import search as rag_search
from .sectionize import sectionize_video
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


@app.command()
def sync():
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


@app.command()
def fetch(
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
        """Fetch transcript for a single video. Returns (video, result, error)."""
        nonlocal interrupted
        if interrupted:
            return (video, "skipped", None)
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
        task = progress.add_task("Fetching", total=len(pending))

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_video, v): v for v in pending}

                for future in as_completed(futures):
                    if interrupted:
                        break

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


@app.command()
def process(
    video_id: str = typer.Argument(None, help="Video ID to process (or all if omitted)"),
    limit: int = typer.Option(None, "-l", "--limit", help="Max videos to process"),
    sectionize_only: bool = typer.Option(False, "--sectionize", help="Only run sectionization"),
    summarize_only: bool = typer.Option(False, "--summarize", help="Only run summarization"),
    model: str = typer.Option(DEFAULT_CHAT_MODEL, "-m", "--model", help="Chat model to use"),
    force: bool = typer.Option(False, "--force", help="Re-process even if already done"),
):
    """Process videos: sectionize and summarize transcripts using GPT."""
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
                        sectionize_video(video.id, db, model)
                        sectionized += 1
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

    console.print(
        f"[green]✓[/green] Processed {len(videos)} videos: "
        f"{sectionized} sectionized, {summarized} summarized, {errors} errors"
    )
    db.close()


@app.command()
def embed(
    video_id: str = typer.Argument(None, help="Video ID to embed (or all if omitted)"),
    model: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "-m", "--model", help="Embedding model"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild entire index"),
    force: bool = typer.Option(False, "--force", help="Re-embed existing sections"),
):
    """Embed sections into FAISS vector index for search."""
    db = get_db()

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
            result = embed_video(video_id, db, model, force)

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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if rebuild:
                progress.add_task("Rebuilding index...", total=None)
            else:
                progress.add_task("Embedding new sections...", total=None)

            result = embed_all_sections(db, model, rebuild=rebuild)

        if result.sections_embedded > 0:
            console.print(
                f"[green]✓[/green] Embedded {result.sections_embedded} sections "
                f"({result.tokens_used} tokens)"
            )
        else:
            console.print("[green]✓[/green] All sections already embedded")

    # Show index stats
    idx_stats = get_index_stats()
    console.print(f"Index: {idx_stats['total_vectors']} vectors")

    db.close()


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "-k", "--top-k", help="Number of sources to retrieve"),
    video: str = typer.Option(None, "-v", "--video", help="Filter to specific video ID"),
    channel: str = typer.Option(None, "-c", "--channel", help="Filter to specific channel ID"),
    no_answer: bool = typer.Option(False, "--no-answer", help="Skip answer generation"),
):
    """Ask a question about video content using RAG."""
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
    console.print(
        f"\n[dim]Latency: {result.latency_ms}ms | "
        f"Tokens: {result.tokens_embedding} embed + {result.tokens_chat} chat[/dim]"
    )

    db.close()


@app.command()
def version():
    """Show version."""
    console.print(f"yt-rag {__version__}")


if __name__ == "__main__":
    app()
