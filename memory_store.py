"""
memory_store.py
----------------
Lightweight memory component (Section 11 of proposal).
Uses SQLite so it works out-of-the-box on Hugging Face Spaces with
zero external database setup. Stores structured prediction results
and lets the orchestrator/LLM retrieve recent relevant context.

This is NOT a medical record system. It only retains what the
orchestrator itself produced (predictions, probabilities, SHAP
summaries, timestamps) for session continuity and explanation
grounding — never a source of clinical truth.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from contextlib import contextmanager


class MemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ckd_prediction TEXT,
                    ckd_probability REAL,
                    ckd_shap_json TEXT,
                    diabetes_prediction TEXT,
                    diabetes_probability REAL,
                    diabetes_shap_json TEXT,
                    errors_json TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON predictions(session_id)
            """)

    def save(self, result: dict, session_id: str = "default") -> str:
        """Persist one orchestrator run. Returns the record id."""
        record_id = str(uuid.uuid4())

        ckd = result.get("ckd") or {}
        diabetes = result.get("diabetes") or {}

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO predictions (
                    id, session_id, timestamp,
                    ckd_prediction, ckd_probability, ckd_shap_json,
                    diabetes_prediction, diabetes_probability, diabetes_shap_json,
                    errors_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id,
                session_id,
                result.get("timestamp", datetime.utcnow().isoformat()),
                ckd.get("prediction"),
                ckd.get("probability"),
                json.dumps(ckd.get("shap")) if ckd.get("shap") else None,
                diabetes.get("prediction"),
                diabetes.get("probability"),
                json.dumps(diabetes.get("shap")) if diabetes.get("shap") else None,
                json.dumps(result.get("errors", [])),
            ))
        return record_id

    def get_recent(self, session_id: str = "default", limit: int = 5) -> list:
        """Retrieve the most recent N records for a session, most recent first."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM predictions
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit)).fetchall()

        records = []
        for row in rows:
            record = dict(row)
            if record.get("ckd_shap_json"):
                record["ckd_shap"] = json.loads(record["ckd_shap_json"])
            if record.get("diabetes_shap_json"):
                record["diabetes_shap"] = json.loads(record["diabetes_shap_json"])
            records.append(record)
        return records

    def get_last(self, session_id: str = "default") -> dict | None:
        recent = self.get_recent(session_id, limit=1)
        return recent[0] if recent else None

    def clear_session(self, session_id: str = "default"):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM predictions WHERE session_id = ?", (session_id,))
