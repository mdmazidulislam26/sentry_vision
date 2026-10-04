"""
Tests for database.py. Uses a temp DB file per test (via monkeypatching
config.DB_PATH) so tests never touch a real sentryvision.db.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentryvision import config, database  # noqa: E402


def test_get_connection_creates_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))

    conn = database.get_connection()
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    conn.close()

    assert "attendance_log" in tables
    assert "security_alerts" in tables


def test_log_attendance_and_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))

    conn = database.get_connection()
    cursor = conn.cursor()
    database.log_attendance(cursor, "Mazidul", track_id=1, timestamp="2026-10-04 10:00:00")
    conn.commit()
    conn.close()

    rows = database.fetch_attendance()
    assert len(rows) == 1
    assert rows[0][0] == "Mazidul"
    assert rows[0][1] == 1


def test_log_alert_and_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))

    conn = database.get_connection()
    cursor = conn.cursor()
    database.log_alert(cursor, track_id=7, timestamp="2026-10-04 10:05:00", snapshot_path="/tmp/x.jpg")
    conn.commit()
    conn.close()

    rows = database.fetch_alerts()
    assert len(rows) == 1
    assert rows[0][0] == 7
    assert rows[0][2] == "/tmp/x.jpg"


def test_fetch_attendance_empty_when_no_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "empty.db"))
    assert database.fetch_attendance() == []
    assert database.fetch_alerts() == []
