# ==============================================================================
#  TETRANETICS — SQLite persistence layer
#  - Stdlib only (sqlite3 + json + os + threading)
#  - One connection per call (sqlite3 connections are not safe across threads)
#  - Used from main.py for: agent identity, session persistence, history queries
# ==============================================================================

import os
import json
import sqlite3
import threading
from typing import Any, Optional

DB_PATH = os.environ.get("TETRA_DB", os.path.join(os.path.dirname(__file__), "tetranetics.db"))

_SCHEMA_VERSION = 1
_init_lock = threading.Lock()
_initialized = False


def _get_conn() -> sqlite3.Connection:
    """One connection per call. Caller must close (or use `with` block)."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db() -> None:
    """Create schema if missing. Idempotent. Call once on app startup."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        with _get_conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id      TEXT PRIMARY KEY,
                    display_name  TEXT,
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id              TEXT PRIMARY KEY,
                    agent_id                TEXT NOT NULL REFERENCES agents(agent_id),
                    scenario_persona        TEXT,
                    scenario_issue          TEXT,
                    difficulty              INTEGER,
                    outcome                 TEXT,
                    total_turns             INTEGER,
                    average_score           REAL,
                    trend                   TEXT,
                    professionalism         INTEGER,
                    customer_satisfaction   INTEGER,
                    problem_resolution      INTEGER,
                    empathy                 INTEGER,
                    communication_clarity   INTEGER,
                    best_turn_score         INTEGER,
                    worst_turn_score        INTEGER,
                    best_turn_agent         TEXT,
                    worst_turn_agent        TEXT,
                    report_text             TEXT,
                    points_json             TEXT,
                    turn_log_json           TEXT,
                    created_via             TEXT,
                    parent_session_id       TEXT,
                    schema_version          INTEGER NOT NULL DEFAULT 1,
                    started_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at                TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_agent_ended
                    ON sessions(agent_id, ended_at DESC);
                """
            )
            cur = c.execute("PRAGMA user_version;")
            current_version = cur.fetchone()[0]
            if current_version == 0:
                c.execute(f"PRAGMA user_version = {_SCHEMA_VERSION};")
        _initialized = True
        print(f"[DB] initialized at {DB_PATH} (schema v{_SCHEMA_VERSION})", flush=True)


# ==============================================================================
#  Agent helpers
# ==============================================================================

def normalize_agent_id(raw: str) -> str:
    return (raw or "").strip().lower()


def ensure_agent(agent_id: str) -> dict:
    """
    Idempotent. Inserts agent if missing; bumps last_seen_at on every call.
    Returns {"agent_id", "display_name", "created"}.
    """
    aid = normalize_agent_id(agent_id)
    if not aid:
        raise ValueError("Empty agent_id")

    with _get_conn() as c:
        existed = c.execute(
            "SELECT agent_id FROM agents WHERE agent_id = ?", (aid,)
        ).fetchone()
        if existed:
            c.execute(
                "UPDATE agents SET last_seen_at = CURRENT_TIMESTAMP WHERE agent_id = ?",
                (aid,),
            )
            row = c.execute(
                "SELECT agent_id, display_name FROM agents WHERE agent_id = ?", (aid,)
            ).fetchone()
            return {
                "agent_id": row["agent_id"],
                "display_name": row["display_name"],
                "created": False,
            }
        c.execute(
            "INSERT INTO agents(agent_id) VALUES (?)", (aid,)
        )
        return {"agent_id": aid, "display_name": None, "created": True}


# ==============================================================================
#  Session helpers
# ==============================================================================

def record_session_terminal(
    session_id: str,
    agent_id: str,
    scenario: dict,
    difficulty: int,
    outcome: str,
    turn_log: list,
    created_via: Optional[str],
    parent_session_id: Optional[str],
) -> None:
    """
    Insert a summary row when a session reaches terminal outcome (win/loss).
    Idempotent via INSERT OR IGNORE. No LLM-derived fields here yet
    (parameter scores, report_text fill in later via update_session_report).
    """
    aid = normalize_agent_id(agent_id)
    if not aid:
        return  # no agent attached — silently skip

    scores = [t.get("score", 0) for t in turn_log] if turn_log else []
    avg = round(sum(scores) / len(scores), 1) if scores else 0.0
    best_score = max(scores) if scores else 0
    worst_score = min(scores) if scores else 0
    best_agent = ""
    worst_agent = ""
    if turn_log:
        best_t = max(turn_log, key=lambda t: t.get("score", 0))
        worst_t = min(turn_log, key=lambda t: t.get("score", 0))
        best_agent = best_t.get("agent", "")
        worst_agent = worst_t.get("agent", "")

    with _get_conn() as c:
        c.execute(
            """
            INSERT OR IGNORE INTO sessions (
                session_id, agent_id, scenario_persona, scenario_issue, difficulty,
                outcome, total_turns, average_score,
                best_turn_score, worst_turn_score,
                best_turn_agent, worst_turn_agent,
                turn_log_json, created_via, parent_session_id,
                ended_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP)
            """,
            (
                session_id, aid,
                scenario.get("customer_persona"), scenario.get("issue_type"),
                difficulty,
                outcome, len(turn_log), avg,
                best_score, worst_score,
                best_agent, worst_agent,
                json.dumps(turn_log, ensure_ascii=False),
                created_via, parent_session_id,
            ),
        )


def update_session_report(session_id: str, report: dict) -> None:
    """
    Fill the report-derived columns once the LLM has produced them.
    Safe to call multiple times — each call overwrites.
    """
    params = report.get("parameters") or {}
    best = report.get("best_turn") or {}
    worst = report.get("worst_turn") or {}
    points = report.get("report_points") or []

    with _get_conn() as c:
        c.execute(
            """
            UPDATE sessions SET
                trend                 = COALESCE(?, trend),
                average_score         = COALESCE(?, average_score),
                total_turns           = COALESCE(?, total_turns),
                professionalism       = ?,
                customer_satisfaction = ?,
                problem_resolution    = ?,
                empathy               = ?,
                communication_clarity = ?,
                best_turn_score       = COALESCE(?, best_turn_score),
                worst_turn_score      = COALESCE(?, worst_turn_score),
                best_turn_agent       = COALESCE(?, best_turn_agent),
                worst_turn_agent      = COALESCE(?, worst_turn_agent),
                report_text           = ?,
                points_json           = ?
            WHERE session_id = ?
            """,
            (
                report.get("trend"),
                report.get("average_score"),
                report.get("total_turns"),
                params.get("professionalism"),
                params.get("customer_satisfaction"),
                params.get("problem_resolution"),
                params.get("empathy"),
                params.get("communication_clarity"),
                best.get("score"),
                worst.get("score"),
                best.get("agent"),
                worst.get("agent"),
                report.get("report_text"),
                json.dumps(points, ensure_ascii=False),
                session_id,
            ),
        )


def list_sessions(agent_id: str) -> list:
    """Summary rows for the history page, newest first."""
    aid = normalize_agent_id(agent_id)
    if not aid:
        return []
    with _get_conn() as c:
        rows = c.execute(
            """
            SELECT session_id, scenario_persona, scenario_issue, difficulty,
                   outcome, total_turns, average_score, trend,
                   professionalism, customer_satisfaction, problem_resolution,
                   empathy, communication_clarity,
                   best_turn_score, worst_turn_score,
                   created_via, parent_session_id,
                   started_at, ended_at
            FROM sessions
            WHERE agent_id = ?
            ORDER BY ended_at DESC, started_at DESC
            """,
            (aid,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_full(agent_id: str, session_id: str) -> Optional[dict]:
    """Full row for replay; turn_log_json + points_json parsed for convenience."""
    aid = normalize_agent_id(agent_id)
    with _get_conn() as c:
        row = c.execute(
            "SELECT * FROM sessions WHERE agent_id = ? AND session_id = ?",
            (aid, session_id),
        ).fetchone()
    if not row:
        return None
    out = dict(row)
    try:
        out["turn_log"] = json.loads(out.get("turn_log_json") or "[]")
    except Exception:
        out["turn_log"] = []
    try:
        out["points"] = json.loads(out.get("points_json") or "[]")
    except Exception:
        out["points"] = []
    return out
