"""SQLite database operations for yt-rag."""

import sqlite3
from datetime import datetime
from pathlib import Path

from .config import DB_PATH, ensure_data_dir
from .models import Channel, Segment, Video

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    last_synced_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    channel_id TEXT,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMP,
    duration_seconds INTEGER,
    transcript_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (channel_id) REFERENCES channels(id)
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    start_time REAL NOT NULL,
    duration REAL NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id, seq);
CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(transcript_status);
"""


class Database:
    """SQLite database wrapper."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            ensure_data_dir()
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def init(self) -> None:
        """Initialize database schema."""
        conn = self.connect()
        conn.executescript(SCHEMA)
        conn.commit()

    def add_channel(self, channel: Channel) -> None:
        """Add or update a channel."""
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO channels (id, name, url, last_synced_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                url = excluded.url,
                last_synced_at = excluded.last_synced_at
            """,
            (channel.id, channel.name, channel.url, channel.last_synced_at),
        )
        conn.commit()

    def get_channel(self, channel_id: str) -> Channel | None:
        """Get a channel by ID."""
        conn = self.connect()
        row = conn.execute("SELECT * FROM channels WHERE id = ?", (channel_id,)).fetchone()
        if row:
            return Channel(**dict(row))
        return None

    def list_channels(self) -> list[Channel]:
        """List all channels."""
        conn = self.connect()
        rows = conn.execute("SELECT * FROM channels").fetchall()
        return [Channel(**dict(row)) for row in rows]

    def update_channel_sync_time(self, channel_id: str) -> None:
        """Update last_synced_at for a channel."""
        conn = self.connect()
        conn.execute(
            "UPDATE channels SET last_synced_at = ? WHERE id = ?",
            (datetime.now(), channel_id),
        )
        conn.commit()

    def add_video(self, video: Video) -> None:
        """Add or update a video."""
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO videos (id, channel_id, title, url, published_at,
                               duration_seconds, transcript_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                duration_seconds = excluded.duration_seconds
            """,
            (
                video.id,
                video.channel_id,
                video.title,
                video.url,
                video.published_at,
                video.duration_seconds,
                video.transcript_status,
                video.created_at or datetime.now(),
            ),
        )
        conn.commit()

    def add_videos(self, videos: list[Video]) -> int:
        """Add multiple videos, return count of new videos added."""
        if not videos:
            return 0
        conn = self.connect()
        video_ids = [v.id for v in videos]
        placeholders = ",".join("?" * len(video_ids))
        existing = {
            row[0]
            for row in conn.execute(
                f"SELECT id FROM videos WHERE id IN ({placeholders})", video_ids
            ).fetchall()
        }
        added = 0
        for video in videos:
            if video.id not in existing:
                added += 1
            conn.execute(
                """
                INSERT INTO videos (id, channel_id, title, url, published_at,
                                   duration_seconds, transcript_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    duration_seconds = excluded.duration_seconds
                """,
                (
                    video.id,
                    video.channel_id,
                    video.title,
                    video.url,
                    video.published_at,
                    video.duration_seconds,
                    video.transcript_status,
                    video.created_at or datetime.now(),
                ),
            )
        conn.commit()
        return added

    def get_video(self, video_id: str) -> Video | None:
        """Get a video by ID."""
        conn = self.connect()
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        if row:
            return Video(**dict(row))
        return None

    def list_videos(self, channel_id: str | None = None, status: str | None = None) -> list[Video]:
        """List videos with optional filters."""
        conn = self.connect()
        query = "SELECT * FROM videos WHERE 1=1"
        params: list = []

        if channel_id:
            query += " AND channel_id = ?"
            params.append(channel_id)
        if status:
            query += " AND transcript_status = ?"
            params.append(status)

        query += " ORDER BY published_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [Video(**dict(row)) for row in rows]

    def get_pending_videos(self) -> list[Video]:
        """Get videos that need transcript fetching."""
        return self.list_videos(status="pending")

    def update_video_status(self, video_id: str, status: str) -> None:
        """Update transcript status for a video."""
        conn = self.connect()
        conn.execute(
            "UPDATE videos SET transcript_status = ? WHERE id = ?",
            (status, video_id),
        )
        conn.commit()

    def add_segments(self, segments: list[Segment]) -> None:
        """Add transcript segments for a video."""
        if not segments:
            return
        conn = self.connect()
        # Delete existing segments for this video
        conn.execute("DELETE FROM segments WHERE video_id = ?", (segments[0].video_id,))
        conn.executemany(
            """
            INSERT INTO segments (video_id, seq, start_time, duration, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(s.video_id, s.seq, s.start_time, s.duration, s.text) for s in segments],
        )
        conn.commit()

    def get_segments(self, video_id: str) -> list[Segment]:
        """Get all segments for a video."""
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM segments WHERE video_id = ? ORDER BY seq", (video_id,)
        ).fetchall()
        return [Segment(**dict(row)) for row in rows]

    def get_full_text(self, video_id: str) -> str:
        """Get full transcript text for a video."""
        segments = self.get_segments(video_id)
        return " ".join(s.text for s in segments)

    def get_stats(self) -> dict:
        """Get database statistics."""
        conn = self.connect()
        stats = {
            "channels": conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0],
            "videos_total": conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
            "videos_fetched": conn.execute(
                "SELECT COUNT(*) FROM videos WHERE transcript_status = 'fetched'"
            ).fetchone()[0],
            "videos_pending": conn.execute(
                "SELECT COUNT(*) FROM videos WHERE transcript_status = 'pending'"
            ).fetchone()[0],
            "videos_unavailable": conn.execute(
                "SELECT COUNT(*) FROM videos WHERE transcript_status = 'unavailable'"
            ).fetchone()[0],
            "segments": conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0],
        }
        return stats
