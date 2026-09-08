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
import os
from datetime import datetime
from contextlib import contextmanager


class MemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.database_url = os.getenv("DATABASE_URL")
        self._init_db()

    @contextmanager
    def _get_conn(self):
        if self.database_url:
            import psycopg
            conn = psycopg.connect(self.database_url)
        else:
            conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            timestamp_type = "TIMESTAMPTZ" if self.database_url else "TEXT"
            probability_type = "DOUBLE PRECISION" if self.database_url else "REAL"
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp {timestamp_type} NOT NULL,
                    ckd_prediction TEXT,
                    ckd_probability {probability_type},
                    ckd_shap_json TEXT,
                    diabetes_prediction TEXT,
                    diabetes_probability {probability_type},
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
            placeholder = "%s" if self.database_url else "?"
            placeholders = ", ".join([placeholder] * 10)
            conn.execute(f"""
                INSERT INTO predictions (
                    id, session_id, timestamp,
                    ckd_prediction, ckd_probability, ckd_shap_json,
                    diabetes_prediction, diabetes_probability, diabetes_shap_json,
                    errors_json
                ) VALUES ({placeholders})
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
            if not self.database_url:
                conn.row_factory = sqlite3.Row
            placeholder = "%s" if self.database_url else "?"
            rows = conn.execute(f"""
                SELECT * FROM predictions
                WHERE session_id = {placeholder}
                ORDER BY timestamp DESC
                LIMIT {placeholder}
            """, (session_id, limit)).fetchall()

        records = []
        for row in rows:
            if self.database_url:
                record = dict(zip(
                    ["id", "session_id", "timestamp", "ckd_prediction", "ckd_probability",
                     "ckd_shap_json", "diabetes_prediction", "diabetes_probability",
                     "diabetes_shap_json", "errors_json"], row
                ))
            else:
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
            placeholder = "%s" if self.database_url else "?"
            conn.execute(f"DELETE FROM predictions WHERE session_id = {placeholder}", (session_id,))
