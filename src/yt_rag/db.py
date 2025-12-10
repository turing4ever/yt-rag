"""SQLite database operations for yt-rag."""

import sqlite3
from datetime import datetime
from pathlib import Path

from .config import DB_PATH, ensure_data_dir
from .models import (
    Channel,
    ChatMessage,
    ChatSession,
    Feedback,
    Keyword,
    QueryLog,
    Section,
    Segment,
    Summary,
    Synonym,
    TestCase,
    Video,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    description TEXT,
    subscriber_count INTEGER,
    tags TEXT,
    handle TEXT,
    category TEXT,
    last_synced_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    channel_id TEXT,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMP,
    duration_seconds INTEGER,
    view_count INTEGER,
    like_count INTEGER,
    comment_count INTEGER,
    description TEXT,
    tags TEXT,
    categories TEXT,
    language TEXT,
    host TEXT,
    guests TEXT,
    availability TEXT,
    transcript_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_refreshed_at TIMESTAMP,
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

CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    start_time REAL,
    end_time REAL,
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE INDEX IF NOT EXISTS idx_sections_video ON sections(video_id, seq);

CREATE TABLE IF NOT EXISTS summaries (
    video_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS query_logs (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    scope_type TEXT,
    scope_id TEXT,
    retrieved_ids TEXT,
    retrieved_scores TEXT,
    answer TEXT,
    latency_ms INTEGER,
    tokens_embedding INTEGER,
    tokens_chat INTEGER,
    model_embedding TEXT,
    model_chat TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feedback (
    query_id TEXT PRIMARY KEY,
    helpful INTEGER,
    source_rating INTEGER,
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES query_logs(id)
);

CREATE TABLE IF NOT EXISTS test_cases (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    scope_type TEXT,
    scope_id TEXT,
    expected_video_ids TEXT,
    expected_keywords TEXT,
    reference_answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    video_id TEXT,
    channel_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL UNIQUE,
    frequency INTEGER DEFAULT 0,
    video_count INTEGER DEFAULT 0,
    category TEXT,
    is_stopword INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_keywords_frequency ON keywords(frequency DESC);

CREATE TABLE IF NOT EXISTS synonyms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL,
    synonym TEXT NOT NULL,
    source TEXT DEFAULT 'manual',
    approved INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(keyword, synonym)
);

CREATE INDEX IF NOT EXISTS idx_synonyms_keyword ON synonyms(keyword);
CREATE INDEX IF NOT EXISTS idx_synonyms_approved ON synonyms(approved);

CREATE TABLE IF NOT EXISTS system_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
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

    def __enter__(self) -> "Database":
        """Context manager entry - initializes if needed."""
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes connection."""
        self.close()

    def init(self) -> None:
        """Initialize database schema."""
        conn = self.connect()
        conn.executescript(SCHEMA)
        conn.commit()
        self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Run database migrations for schema updates."""
        # Migrate videos table
        cursor = conn.execute("PRAGMA table_info(videos)")
        video_cols = {row[1] for row in cursor.fetchall()}

        video_new_columns = [
            ("view_count", "INTEGER"),
            ("like_count", "INTEGER"),
            ("comment_count", "INTEGER"),
            ("description", "TEXT"),
            ("tags", "TEXT"),
            ("categories", "TEXT"),
            ("language", "TEXT"),
            ("host", "TEXT"),
            ("guests", "TEXT"),
            ("availability", "TEXT"),
            ("metadata_refreshed_at", "TIMESTAMP"),
        ]

        for col_name, col_type in video_new_columns:
            if col_name not in video_cols:
                conn.execute(f"ALTER TABLE videos ADD COLUMN {col_name} {col_type}")

        # Migrate channels table
        cursor = conn.execute("PRAGMA table_info(channels)")
        channel_cols = {row[1] for row in cursor.fetchall()}

        channel_new_columns = [
            ("description", "TEXT"),
            ("subscriber_count", "INTEGER"),
            ("tags", "TEXT"),
            ("handle", "TEXT"),
        ]

        for col_name, col_type in channel_new_columns:
            if col_name not in channel_cols:
                conn.execute(f"ALTER TABLE channels ADD COLUMN {col_name} {col_type}")

        conn.commit()

    def add_channel(self, channel: Channel) -> None:
        """Add or update a channel."""
        import json

        conn = self.connect()
        tags_json = json.dumps(channel.tags) if channel.tags else None
        conn.execute(
            """
            INSERT INTO channels (
                id, name, url, description, subscriber_count,
                tags, handle, category, last_synced_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                url = excluded.url,
                description = COALESCE(excluded.description, channels.description),
                subscriber_count = COALESCE(excluded.subscriber_count, channels.subscriber_count),
                tags = COALESCE(excluded.tags, channels.tags),
                handle = COALESCE(excluded.handle, channels.handle),
                category = COALESCE(excluded.category, channels.category),
                last_synced_at = excluded.last_synced_at
            """,
            (
                channel.id,
                channel.name,
                channel.url,
                channel.description,
                channel.subscriber_count,
                tags_json,
                channel.handle,
                channel.category,
                channel.last_synced_at,
            ),
        )
        conn.commit()

    def _channel_from_row(self, row) -> Channel:
        """Convert a database row to a Channel, deserializing JSON fields."""
        import json

        d = dict(row)
        if d.get("tags"):
            d["tags"] = json.loads(d["tags"])
        return Channel(**d)

    def get_channel(self, channel_id: str) -> Channel | None:
        """Get a channel by ID."""
        conn = self.connect()
        row = conn.execute("SELECT * FROM channels WHERE id = ?", (channel_id,)).fetchone()
        if row:
            return self._channel_from_row(row)
        return None

    def list_channels(self) -> list[Channel]:
        """List all channels."""
        conn = self.connect()
        rows = conn.execute("SELECT * FROM channels").fetchall()
        return [self._channel_from_row(row) for row in rows]

    def update_channel_sync_time(self, channel_id: str) -> None:
        """Update last_synced_at for a channel."""
        conn = self.connect()
        conn.execute(
            "UPDATE channels SET last_synced_at = ? WHERE id = ?",
            (datetime.now(), channel_id),
        )
        conn.commit()

    def set_channel_category(self, channel_id: str, category: str) -> None:
        """Set the category for a channel."""
        conn = self.connect()
        conn.execute(
            "UPDATE channels SET category = ? WHERE id = ?",
            (category, channel_id),
        )
        conn.commit()

    def add_video(self, video: Video, update_metadata_ts: bool = True) -> None:
        """Add or update a video.

        Args:
            video: Video to add/update
            update_metadata_ts: If True, set metadata_refreshed_at to now when metadata is present
        """
        import json

        conn = self.connect()
        tags_json = json.dumps(video.tags) if video.tags else None
        categories_json = json.dumps(video.categories) if video.categories else None
        guests_json = json.dumps(video.guests) if video.guests else None

        # Set metadata_refreshed_at if we have metadata and flag is set
        metadata_ts = None
        if update_metadata_ts and video.description:
            metadata_ts = datetime.now()

        conn.execute(
            """
            INSERT INTO videos (id, channel_id, title, url, published_at,
                               duration_seconds, view_count, like_count, comment_count,
                               description, tags, categories, language, host, guests,
                               availability, transcript_status, created_at, metadata_refreshed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                published_at = COALESCE(excluded.published_at, videos.published_at),
                duration_seconds = excluded.duration_seconds,
                view_count = COALESCE(excluded.view_count, videos.view_count),
                like_count = COALESCE(excluded.like_count, videos.like_count),
                comment_count = COALESCE(excluded.comment_count, videos.comment_count),
                description = COALESCE(excluded.description, videos.description),
                tags = COALESCE(excluded.tags, videos.tags),
                categories = COALESCE(excluded.categories, videos.categories),
                language = COALESCE(excluded.language, videos.language),
                host = COALESCE(excluded.host, videos.host),
                guests = COALESCE(excluded.guests, videos.guests),
                availability = COALESCE(excluded.availability, videos.availability),
                metadata_refreshed_at = COALESCE(
                    excluded.metadata_refreshed_at, videos.metadata_refreshed_at
                )
            """,
            (
                video.id,
                video.channel_id,
                video.title,
                video.url,
                video.published_at,
                video.duration_seconds,
                video.view_count,
                video.like_count,
                video.comment_count,
                video.description,
                tags_json,
                categories_json,
                video.language,
                video.host,
                guests_json,
                video.availability,
                video.transcript_status,
                video.created_at or datetime.now(),
                metadata_ts,
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

    def _parse_video_row(self, row: sqlite3.Row) -> Video:
        """Parse a video row, deserializing JSON fields."""
        import json

        data = dict(row)
        # Parse JSON fields
        for field in ("tags", "categories", "guests"):
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except (json.JSONDecodeError, TypeError):
                    data[field] = None
        return Video(**data)

    def get_video(self, video_id: str) -> Video | None:
        """Get a video by ID."""
        conn = self.connect()
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        if row:
            return self._parse_video_row(row)
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
        return [self._parse_video_row(row) for row in rows]

    def get_top_videos_by_views(
        self,
        limit: int = 10,
        channel_id: str | None = None,
        title_contains: str | None = None,
        published_after: datetime | None = None,
    ) -> list[Video]:
        """Get top videos sorted by view count.

        Args:
            limit: Max number of videos to return
            channel_id: Filter to specific channel
            title_contains: Filter by title substring (case-insensitive)
            published_after: Only include videos published after this date

        Returns:
            List of videos sorted by view_count descending
        """
        conn = self.connect()
        query = "SELECT * FROM videos WHERE view_count IS NOT NULL"
        params: list = []

        if channel_id:
            query += " AND channel_id = ?"
            params.append(channel_id)

        if title_contains:
            query += " AND LOWER(title) LIKE ?"
            params.append(f"%{title_contains.lower()}%")

        if published_after:
            query += " AND published_at >= ?"
            params.append(published_after.isoformat())

        query += " ORDER BY view_count DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._parse_video_row(row) for row in rows]

    def search_videos(
        self,
        channel_id: str | None = None,
        status: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        min_duration: int | None = None,
        max_duration: int | None = None,
        host: str | None = None,
        guest: str | None = None,
        min_sections: int | None = None,
        max_sections: int | None = None,
        title_contains: str | None = None,
        order_by: str = "published_at",
        order_desc: bool = True,
        limit: int | None = None,
    ) -> list[tuple[Video, int]]:
        """Search videos with metadata filters.

        Args:
            channel_id: Filter by channel
            status: Filter by transcript status
            after: Published after this date
            before: Published before this date
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            host: Filter by host (case-insensitive substring)
            guest: Filter by guest name (case-insensitive substring)
            min_sections: Minimum number of sections
            max_sections: Maximum number of sections
            title_contains: Title contains this string (case-insensitive)
            order_by: Column to sort by (published_at, duration_seconds, title)
            order_desc: Sort descending if True
            limit: Max results to return

        Returns:
            List of (Video, section_count) tuples
        """
        conn = self.connect()

        # Build query with section count
        query = """
            SELECT v.*, COALESCE(s.section_count, 0) as section_count
            FROM videos v
            LEFT JOIN (
                SELECT video_id, COUNT(*) as section_count
                FROM sections
                GROUP BY video_id
            ) s ON v.id = s.video_id
            WHERE 1=1
        """
        params: list = []

        if channel_id:
            query += " AND v.channel_id = ?"
            params.append(channel_id)
        if status:
            query += " AND v.transcript_status = ?"
            params.append(status)
        if after:
            query += " AND v.published_at >= ?"
            params.append(after)
        if before:
            query += " AND v.published_at <= ?"
            params.append(before)
        if min_duration is not None:
            query += " AND v.duration_seconds >= ?"
            params.append(min_duration)
        if max_duration is not None:
            query += " AND v.duration_seconds <= ?"
            params.append(max_duration)
        if host:
            query += " AND LOWER(v.host) LIKE ?"
            params.append(f"%{host.lower()}%")
        if guest:
            query += " AND LOWER(v.guests) LIKE ?"
            params.append(f"%{guest.lower()}%")
        if min_sections is not None:
            query += " AND COALESCE(s.section_count, 0) >= ?"
            params.append(min_sections)
        if max_sections is not None:
            query += " AND COALESCE(s.section_count, 0) <= ?"
            params.append(max_sections)
        if title_contains:
            query += " AND LOWER(v.title) LIKE ?"
            params.append(f"%{title_contains.lower()}%")

        # Order by
        valid_order = {"published_at", "duration_seconds", "title", "section_count"}
        if order_by not in valid_order:
            order_by = "published_at"
        direction = "DESC" if order_desc else "ASC"

        if order_by == "section_count":
            query += f" ORDER BY section_count {direction}"
        else:
            query += f" ORDER BY v.{order_by} {direction}"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            section_count = d.pop("section_count")
            video = self._parse_video_row_dict(d)
            results.append((video, section_count))
        return results

    def _parse_video_row_dict(self, data: dict) -> Video:
        """Parse a video dict, deserializing JSON fields."""
        import json

        for field in ("tags", "categories", "guests"):
            if data.get(field):
                try:
                    data[field] = json.loads(data[field])
                except (json.JSONDecodeError, TypeError):
                    data[field] = None
        return Video(**data)

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
            "sections": conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0],
            "summaries": conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0],
        }
        return stats

    # System info methods

    def get_system_info(self, key: str) -> str | None:
        """Get a system info value by key."""
        conn = self.connect()
        row = conn.execute(
            "SELECT value FROM system_info WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_system_info(self, key: str, value: str) -> None:
        """Set a system info value (insert or update)."""
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO system_info (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """,
            (key, value),
        )
        conn.commit()

    def get_all_system_info(self) -> dict[str, str]:
        """Get all system info as a dictionary."""
        conn = self.connect()
        rows = conn.execute("SELECT key, value FROM system_info").fetchall()
        return {row[0]: row[1] for row in rows}

    # Section methods

    def add_sections(self, sections: list[Section]) -> None:
        """Add sections for a video, replacing existing ones."""
        if not sections:
            return
        conn = self.connect()
        video_id = sections[0].video_id
        conn.execute("DELETE FROM sections WHERE video_id = ?", (video_id,))
        conn.executemany(
            """
            INSERT INTO sections (id, video_id, seq, title, content,
                                 start_time, end_time, word_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.id,
                    s.video_id,
                    s.seq,
                    s.title,
                    s.content,
                    s.start_time,
                    s.end_time,
                    s.word_count or len(s.content.split()),
                    s.created_at or datetime.now(),
                )
                for s in sections
            ],
        )
        conn.commit()

    def get_sections(self, video_id: str) -> list[Section]:
        """Get all sections for a video."""
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM sections WHERE video_id = ? ORDER BY seq", (video_id,)
        ).fetchall()
        return [Section(**dict(row)) for row in rows]

    def get_section(self, section_id: str) -> Section | None:
        """Get a section by ID."""
        conn = self.connect()
        row = conn.execute("SELECT * FROM sections WHERE id = ?", (section_id,)).fetchone()
        if row:
            return Section(**dict(row))
        return None

    def get_all_sections(self) -> list[Section]:
        """Get all sections across all videos."""
        conn = self.connect()
        rows = conn.execute("SELECT * FROM sections ORDER BY video_id, seq").fetchall()
        return [Section(**dict(row)) for row in rows]

    def get_sections_by_ids(self, section_ids: list[str]) -> list[Section]:
        """Get sections by their IDs."""
        if not section_ids:
            return []
        conn = self.connect()
        placeholders = ",".join("?" * len(section_ids))
        rows = conn.execute(
            f"SELECT * FROM sections WHERE id IN ({placeholders})", section_ids
        ).fetchall()
        return [Section(**dict(row)) for row in rows]

    # Summary methods

    def add_summary(self, summary: Summary) -> None:
        """Add or update a video summary."""
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO summaries (video_id, summary, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                summary = excluded.summary,
                created_at = excluded.created_at
            """,
            (summary.video_id, summary.summary, summary.created_at or datetime.now()),
        )
        conn.commit()

    def get_summary(self, video_id: str) -> Summary | None:
        """Get summary for a video."""
        conn = self.connect()
        row = conn.execute("SELECT * FROM summaries WHERE video_id = ?", (video_id,)).fetchone()
        if row:
            return Summary(**dict(row))
        return None

    def get_all_summaries(self) -> list[Summary]:
        """Get all summaries."""
        conn = self.connect()
        rows = conn.execute("SELECT * FROM summaries").fetchall()
        return [Summary(**dict(row)) for row in rows]

    # Query log methods

    def add_query_log(self, log: QueryLog) -> None:
        """Add a query log entry."""
        import json

        conn = self.connect()
        conn.execute(
            """
            INSERT INTO query_logs (id, query, scope_type, scope_id, retrieved_ids,
                                   retrieved_scores, answer, latency_ms, tokens_embedding,
                                   tokens_chat, model_embedding, model_chat, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log.id,
                log.query,
                log.scope_type,
                log.scope_id,
                json.dumps(log.retrieved_ids) if log.retrieved_ids else None,
                json.dumps(log.retrieved_scores) if log.retrieved_scores else None,
                log.answer,
                log.latency_ms,
                log.tokens_embedding,
                log.tokens_chat,
                log.model_embedding,
                log.model_chat,
                log.created_at or datetime.now(),
            ),
        )
        conn.commit()

    def get_query_logs(self, limit: int = 50) -> list[QueryLog]:
        """Get recent query logs."""
        import json

        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM query_logs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        logs = []
        for row in rows:
            d = dict(row)
            if d.get("retrieved_ids"):
                d["retrieved_ids"] = json.loads(d["retrieved_ids"])
            if d.get("retrieved_scores"):
                d["retrieved_scores"] = json.loads(d["retrieved_scores"])
            logs.append(QueryLog(**d))
        return logs

    # Feedback methods

    def add_feedback(self, feedback: Feedback) -> None:
        """Add feedback for a query."""
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO feedback (query_id, helpful, source_rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(query_id) DO UPDATE SET
                helpful = excluded.helpful,
                source_rating = excluded.source_rating,
                comment = excluded.comment
            """,
            (
                feedback.query_id,
                1 if feedback.helpful else (0 if feedback.helpful is False else None),
                feedback.source_rating,
                feedback.comment,
                feedback.created_at or datetime.now(),
            ),
        )
        conn.commit()

    # Test case methods

    def add_test_case(self, test: TestCase) -> None:
        """Add a test case."""
        import json

        conn = self.connect()
        conn.execute(
            """
            INSERT INTO test_cases (id, query, scope_type, scope_id, expected_video_ids,
                                   expected_keywords, reference_answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                query = excluded.query,
                expected_video_ids = excluded.expected_video_ids,
                expected_keywords = excluded.expected_keywords,
                reference_answer = excluded.reference_answer
            """,
            (
                test.id,
                test.query,
                test.scope_type,
                test.scope_id,
                json.dumps(test.expected_video_ids) if test.expected_video_ids else None,
                json.dumps(test.expected_keywords) if test.expected_keywords else None,
                test.reference_answer,
                test.created_at or datetime.now(),
            ),
        )
        conn.commit()

    def get_test_cases(self) -> list[TestCase]:
        """Get all test cases."""
        import json

        conn = self.connect()
        rows = conn.execute("SELECT * FROM test_cases").fetchall()
        cases = []
        for row in rows:
            d = dict(row)
            if d.get("expected_video_ids"):
                d["expected_video_ids"] = json.loads(d["expected_video_ids"])
            if d.get("expected_keywords"):
                d["expected_keywords"] = json.loads(d["expected_keywords"])
            cases.append(TestCase(**d))
        return cases

    # Chat session methods

    def create_chat_session(
        self,
        session_id: str,
        title: str,
        video_id: str | None = None,
        channel_id: str | None = None,
    ) -> ChatSession:
        """Create a new chat session."""
        conn = self.connect()
        now = datetime.now()
        conn.execute(
            """
            INSERT INTO chat_sessions (id, title, video_id, channel_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, title, video_id, channel_id, now, now),
        )
        conn.commit()
        return ChatSession(
            id=session_id,
            title=title,
            video_id=video_id,
            channel_id=channel_id,
            created_at=now,
            updated_at=now,
        )

    def get_chat_session(self, session_id: str) -> ChatSession | None:
        """Get a chat session by ID."""
        conn = self.connect()
        row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        if row:
            return ChatSession(**dict(row))
        return None

    def list_chat_sessions(self, limit: int = 20) -> list[ChatSession]:
        """List recent chat sessions, most recent first."""
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [ChatSession(**dict(row)) for row in rows]

    def update_chat_session_title(self, session_id: str, title: str) -> None:
        """Update a session's title."""
        conn = self.connect()
        conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, datetime.now(), session_id),
        )
        conn.commit()

    def touch_chat_session(self, session_id: str) -> None:
        """Update session's updated_at timestamp."""
        conn = self.connect()
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (datetime.now(), session_id),
        )
        conn.commit()

    def delete_chat_session(self, session_id: str) -> None:
        """Delete a chat session and all its messages."""
        conn = self.connect()
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        conn.commit()

    def add_chat_message(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> ChatMessage:
        """Add a message to a chat session."""
        conn = self.connect()
        now = datetime.now()
        conn.execute(
            """
            INSERT INTO chat_messages (id, session_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, session_id, role, content, now),
        )
        # Update session's updated_at
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        conn.commit()
        return ChatMessage(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            created_at=now,
        )

    def get_chat_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Get messages for a session, oldest first."""
        conn = self.connect()
        if limit:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [ChatMessage(**dict(row)) for row in rows]

    def get_chat_message_count(self, session_id: str) -> int:
        """Get count of messages in a session."""
        conn = self.connect()
        count = conn.execute(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        return count

    # --- Keyword methods ---

    def upsert_keyword(self, keyword: str, frequency: int = 1, video_count: int = 1) -> None:
        """Insert or update a keyword."""
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO keywords (keyword, frequency, video_count)
            VALUES (?, ?, ?)
            ON CONFLICT(keyword) DO UPDATE SET
                frequency = frequency + excluded.frequency,
                video_count = video_count + excluded.video_count
            """,
            (keyword.lower(), frequency, video_count),
        )
        conn.commit()

    def get_keywords(self, limit: int = 100, min_frequency: int = 1) -> list[Keyword]:
        """Get top keywords by frequency."""
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT * FROM keywords
            WHERE frequency >= ? AND is_stopword = 0
            ORDER BY frequency DESC
            LIMIT ?
            """,
            (min_frequency, limit),
        ).fetchall()
        return [Keyword(**dict(row)) for row in rows]

    def mark_stopword(self, keyword: str) -> None:
        """Mark a keyword as a stopword."""
        conn = self.connect()
        conn.execute(
            "UPDATE keywords SET is_stopword = 1 WHERE keyword = ?",
            (keyword.lower(),),
        )
        conn.commit()

    def set_keyword_category(self, keyword: str, category: str) -> None:
        """Set category for a keyword."""
        conn = self.connect()
        conn.execute(
            "UPDATE keywords SET category = ? WHERE keyword = ?",
            (category, keyword.lower()),
        )
        conn.commit()

    # --- Synonym methods ---

    def add_synonym(
        self, keyword: str, synonym: str, source: str = "manual", approved: bool = False
    ) -> None:
        """Add a synonym mapping."""
        conn = self.connect()
        conn.execute(
            """
            INSERT OR IGNORE INTO synonyms (keyword, synonym, source, approved)
            VALUES (?, ?, ?, ?)
            """,
            (keyword.lower(), synonym.lower(), source, int(approved)),
        )
        conn.commit()

    def add_synonyms_batch(
        self, mappings: list[tuple[str, str]], source: str = "manual", approved: bool = False
    ) -> None:
        """Add multiple synonym mappings."""
        conn = self.connect()
        conn.executemany(
            """
            INSERT OR IGNORE INTO synonyms (keyword, synonym, source, approved)
            VALUES (?, ?, ?, ?)
            """,
            [(kw.lower(), syn.lower(), source, int(approved)) for kw, syn in mappings],
        )
        conn.commit()

    def get_synonyms(self, keyword: str, approved_only: bool = True) -> list[str]:
        """Get synonyms for a keyword."""
        conn = self.connect()
        query = "SELECT synonym FROM synonyms WHERE keyword = ?"
        if approved_only:
            query += " AND approved = 1"
        rows = conn.execute(query, (keyword.lower(),)).fetchall()
        return [row[0] for row in rows]

    def get_all_synonyms(self, approved_only: bool = True) -> dict[str, list[str]]:
        """Get all synonyms as a dict mapping keyword -> [synonyms]."""
        conn = self.connect()
        query = "SELECT keyword, synonym FROM synonyms"
        if approved_only:
            query += " WHERE approved = 1"
        rows = conn.execute(query).fetchall()

        result: dict[str, list[str]] = {}
        for row in rows:
            kw, syn = row[0], row[1]
            if kw not in result:
                result[kw] = []
            result[kw].append(syn)
        return result

    def list_pending_synonyms(self, limit: int = 50) -> list[Synonym]:
        """List synonyms pending approval."""
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM synonyms WHERE approved = 0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Synonym(**dict(row)) for row in rows]

    def approve_synonym(self, keyword: str, synonym: str) -> None:
        """Approve a synonym."""
        conn = self.connect()
        conn.execute(
            "UPDATE synonyms SET approved = 1 WHERE keyword = ? AND synonym = ?",
            (keyword.lower(), synonym.lower()),
        )
        conn.commit()

    def reject_synonym(self, keyword: str, synonym: str) -> None:
        """Reject (delete) a synonym."""
        conn = self.connect()
        conn.execute(
            "DELETE FROM synonyms WHERE keyword = ? AND synonym = ?",
            (keyword.lower(), synonym.lower()),
        )
        conn.commit()

    def remove_synonyms_for_keyword(self, keyword: str) -> int:
        """Remove all synonyms for a keyword.

        Returns:
            Number of synonyms deleted.
        """
        conn = self.connect()
        cursor = conn.execute(
            "DELETE FROM synonyms WHERE keyword = ?",
            (keyword.lower(),),
        )
        conn.commit()
        return cursor.rowcount
