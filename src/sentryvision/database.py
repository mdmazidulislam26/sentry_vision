"""
SQLite persistence layer for attendance logs and security alerts.

Isolating this behind small functions (rather than inlining SQL in the
pipeline loop) means the schema can change, or the backend can later be
swapped for Postgres/MySQL in production, without touching detection logic.
"""

import sqlite3

from . import config


def get_connection():
    """Open a connection and ensure both tables exist."""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            entry_time TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS security_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            detected_time TEXT NOT NULL,
            snapshot_path TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn


def log_attendance(cursor, name, track_id, timestamp):
    cursor.execute(
        "INSERT INTO attendance_log (name, track_id, entry_time) VALUES (?, ?, ?)",
        (name, track_id, timestamp),
    )


def log_alert(cursor, track_id, timestamp, snapshot_path):
    cursor.execute(
        "INSERT INTO security_alerts (track_id, detected_time, snapshot_path) VALUES (?, ?, ?)",
        (track_id, timestamp, snapshot_path),
    )


def fetch_attendance():
    conn = get_connection()
    rows = conn.execute(
        "SELECT name, track_id, entry_time FROM attendance_log ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def fetch_alerts():
    conn = get_connection()
    rows = conn.execute(
        "SELECT track_id, detected_time, snapshot_path FROM security_alerts ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows
