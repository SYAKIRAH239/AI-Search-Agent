import sqlite3
import json
import time
import hashlib
from contextlib import contextmanager

DB_PATH = "search_cache.db"
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours — reddit/search results go stale fast


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_cache():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.commit()


def _make_key(operation, query, **params):
    raw = f"{operation}:{query}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(operation, query, **params):
    key = _make_key(operation, query, **params)
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT result_json, created_at FROM search_cache WHERE cache_key = ?", (key,)
        ).fetchone()

    if not row:
        return None

    result_json, created_at = row
    if time.time() - created_at > CACHE_TTL_SECONDS:
        return None  # stale, treat as a miss

    print(f"Cache hit for {operation}: {query}")
    return json.loads(result_json)


def set_cached(operation, query, result, **params):
    if result is None:
        return  # don't cache failures
    key = _make_key(operation, query, **params)
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO search_cache (cache_key, result_json, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(result), time.time())
        )
        conn.commit()