"""SQLite-backed account storage for the Gradio login flow."""

import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager


class AuthStore:
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @staticmethod
    def _normalize_email(email: str) -> str:
        return (email or "").strip().lower()

    @staticmethod
    def _hash_password(password: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, 120_000
        )
        return f"{salt.hex()}${digest.hex()}"

    @staticmethod
    def _verify_password(password: str, stored: str) -> bool:
        try:
            salt_hex, digest_hex = stored.split("$", 1)
            expected = bytes.fromhex(digest_hex)
            actual = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 120_000
            )
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False

    def create_user(self, email: str, password: str) -> tuple[bool, str]:
        email = self._normalize_email(email)
        if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            return False, "Enter a valid email address."
        if len(password or "") < 8:
            return False, "Password must be at least 8 characters."

        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                    (email, self._hash_password(password)),
                )
        except sqlite3.IntegrityError:
            return False, "An account with that email already exists."
        return True, "Account created. You can log in now."

    def authenticate(self, email: str, password: str) -> tuple[bool, str]:
        email = self._normalize_email(email)
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE email = ?", (email,)
            ).fetchone()
        if row and self._verify_password(password or "", row[0]):
            return True, email
        return False, "Email or password is incorrect."