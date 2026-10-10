"""
Keep the suite off the developer's own run database.

`api.deps` opens DATABASE_URL at import time, and its default is a SQLite
file in the working directory. Point it at a throwaway file before anything
imports the app.
"""

import os
import tempfile
from pathlib import Path

_DB = Path(tempfile.mkdtemp(prefix="quantlab-test-")) / "runs.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_DB.as_posix()}"


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_daily_cap_unless_asked():
    """Settings outlive a test, so a daily cap of 1 set by one test would
    throttle the next file's module client. Tests that exercise the cap set
    it themselves, after this runs."""
    from api import deps
    deps.settings.daily_limit = 0
    deps.settings.user_daily_limit = 0
    yield
