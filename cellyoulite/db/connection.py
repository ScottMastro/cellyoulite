"""Connection helpers: one short-lived connection per request (FastAPI runs
sync handlers in a threadpool, so a single shared connection is unsafe).
WAL mode lets the frequent status-polling reads run alongside writes."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def db_path() -> Path:
    """Location of the SQLite file: $CELLYOULITE_DB or ./cellyoulite.db."""
    env = os.environ.get("CELLYOULITE_DB")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / "cellyoulite.db"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
