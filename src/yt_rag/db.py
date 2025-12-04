"""SQLite database operations for yt-rag."""

import sqlite3
from datetime import datetime
from pathlib import Path

from .config import DB_PATH, ensure_data_dir
from .models import Channel, Feedback, QueryLog, Section, Segment, Summary, TestCase, Video

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
            "sections": conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0],
            "summaries": conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0],
        }
        return stats

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
