import sqlite3
import threading
from functools import lru_cache
from config import get_settings
from cache import explain_cache

_settings = get_settings()
_DB_PATH = _settings.dict_db_path
_local = threading.local()


def _get_conn():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def _ensure_db():
    """Create table/index if missing (idempotent)."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS medical_terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            definition TEXT NOT NULL,
            clinical_significance TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_term_lower 
        ON medical_terms (LOWER(term))
    """)
    conn.commit()


_ensure_db()


def lookup_terms(extracted_terms: list[str]) -> dict:
    """
    Batch lookup against local SQLite.
    Returns: {"found": {term: {...}}, "missing": [terms]}
    """
    if not extracted_terms:
        return {"found": {}, "missing": []}

    conn = _get_conn()
    placeholders = ",".join(["?"] * len(extracted_terms))
    query = f"""
        SELECT term, category, definition, clinical_significance 
        FROM medical_terms 
        WHERE LOWER(term) IN ({','.join(['LOWER(?)'] * len(extracted_terms))})
    """
    rows = conn.execute(query, extracted_terms).fetchall()

    found = {}
    found_lower = set()
    for row in rows:
        term = row["term"]
        found[term] = {
            "category": row["category"],
            "definition": row["definition"],
            "significance": row["clinical_significance"],
            "source": "Local Medical DB",
        }
        found_lower.add(term.lower())

    missing = [t for t in extracted_terms if t.lower() not in found_lower]
    return {"found": found, "missing": missing}


def add_term(term: str, category: str, definition: str, significance: str = "") -> bool:
    """Insert or update a term. Returns True if inserted/updated."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO medical_terms (term, category, definition, clinical_significance)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(term) DO UPDATE SET
                category=excluded.category,
                definition=excluded.definition,
                clinical_significance=excluded.clinical_significance
            """,
            (term, category, definition, significance),
        )
        conn.commit()
        # Invalidate explain cache since new term available
        explain_cache._backend.clear_prefix("explain:")
        return True
    except Exception:
        return False


def bulk_add(terms: list[tuple[str, str, str, str]]) -> int:
    """Bulk insert. Returns count of inserted/updated."""
    conn = _get_conn()
    cur = conn.executemany(
        """
        INSERT INTO medical_terms (term, category, definition, clinical_significance)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(term) DO UPDATE SET
            category=excluded.category,
            definition=excluded.definition,
            clinical_significance=excluded.clinical_significance
        """,
        terms,
    )
    conn.commit()
    if cur.rowcount:
        explain_cache._backend.clear_prefix("explain:")
    return cur.rowcount


def get_all_terms() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT term, category, definition, clinical_significance FROM medical_terms ORDER BY term").fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    test = ["eGFR", "HbA1c", "UnknownMarkerXYZ"]
    resp = lookup_terms(test)
    print("FOUND:", list(resp["found"].keys()))
    print("MISSING:", resp["missing"])