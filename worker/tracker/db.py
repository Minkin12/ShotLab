"""Postgres writes. Re-processing a video replaces its session."""
from __future__ import annotations

from datetime import datetime, timedelta

import psycopg

from .frames import VideoInfo

# Columns added after the first release; safe to run on every save.
MIGRATIONS = [
    "ALTER TABLE shots ADD COLUMN IF NOT EXISTS exit_speed_mps REAL",
    "ALTER TABLE shots ADD COLUMN IF NOT EXISTS make_prob REAL",
]


def save_session(dsn: str, info: VideoInfo, video_file: str, started: datetime, rows: list[dict]) -> int:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for sql in MIGRATIONS:
            cur.execute(sql)
        cur.execute("DELETE FROM sessions WHERE video_file = %s", (video_file,))
        cur.execute(
            """INSERT INTO sessions (video_file, recorded_at, fps, frame_count, duration_s)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (video_file, started, info.fps, info.frame_count, info.duration_s),
        )
        session_id = cur.fetchone()[0]
        if rows:
            cols = ["session_id", "shot_time", *rows[0].keys()]
            sql = f"INSERT INTO shots ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})"
            cur.executemany(
                sql,
                [
                    (session_id, started + timedelta(seconds=r["video_offset_s"]), *r.values())
                    for r in rows
                ],
            )
        conn.commit()
        return session_id
