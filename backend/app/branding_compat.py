"""Session and database naming used by the public application."""
from pathlib import Path
from typing import Mapping

SESSION_COOKIE = "saer_session"

def session_token(cookies: Mapping[str, str]) -> str | None:
    return cookies.get(SESSION_COOKIE)

def database_path(directory: Path) -> Path:
    return directory / "saer.sqlite"
