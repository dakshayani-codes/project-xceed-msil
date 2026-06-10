# backend/main.py — Project Xceed FastAPI backend
# Endpoints:
#   POST /violation   — detect.py calls this on every state change
#   GET  /violations  — Streamlit reads last N events
#   GET  /status      — current live state (latest event)

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import sqlite3
import os

# ── App ──────────────────────────────────────────────
app = FastAPI(title="Project Xceed API", version="1.0")

DB_PATH = os.path.expanduser("~/project-xceed/backend/xceed.db")

# ── DB init ──────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            class_name TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            alert      INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ── Models ───────────────────────────────────────────
class ViolationEvent(BaseModel):
    class_name: str          # 'proper_belt' | 'no_belt' | 'clipped_behind' | 'decoy' | 'none'
    confidence: float        # 0.0–1.0
    alert: bool              # True if temporal filter fired


ALERT_CLASSES = {'no_belt', 'clipped_behind', 'decoy'}

# ── Routes ───────────────────────────────────────────

@app.get("/status")
def get_status():
    """Returns the most recent detection event — used by dashboard for live state card."""
    conn = get_db()
    row  = conn.execute(
        "SELECT * FROM violations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if row is None:
        return {
            "class_name": "none",
            "confidence": 0.0,
            "alert":      False,
            "timestamp":  None,
            "is_violation": False,
        }

    return {
        "class_name":   row["class_name"],
        "confidence":   round(row["confidence"], 3),
        "alert":        bool(row["alert"]),
        "timestamp":    row["timestamp"],
        "is_violation": row["class_name"] in ALERT_CLASSES,
    }


@app.post("/violation")
def post_violation(event: ViolationEvent):
    """Called by detect.py on every detection (or state change)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    conn.execute(
        "INSERT INTO violations (timestamp, class_name, confidence, alert) VALUES (?,?,?,?)",
        (ts, event.class_name, event.confidence, int(event.alert))
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "timestamp": ts}


@app.get("/violations")
def get_violations(limit: int = 50):
    """Returns last N violation events — used by dashboard log table."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM violations ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    return [
        {
            "id":         r["id"],
            "timestamp":  r["timestamp"],
            "class_name": r["class_name"],
            "confidence": round(r["confidence"], 3),
            "alert":      bool(r["alert"]),
        }
        for r in rows
    ]


@app.get("/violations/summary")
def get_summary():
    """Returns count per class — used by dashboard stats."""
    conn = get_db()
    rows = conn.execute(
        "SELECT class_name, COUNT(*) as count FROM violations GROUP BY class_name"
    ).fetchall()
    conn.close()
    return {r["class_name"]: r["count"] for r in rows}


@app.delete("/violations")
def clear_violations():
    """Clears the log — useful before a fresh demo run."""
    conn = get_db()
    conn.execute("DELETE FROM violations")
    conn.commit()
    conn.close()
    return {"status": "cleared"}
