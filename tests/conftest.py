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
    deps.settings.run_retention_days = 30
    deps.settings.visitor_key = ""
    deps.settings.max_body_bytes = 64 * 1024
    # Never a real research provider unless a test installs a fake one.
    deps.settings.anthropic_api_key = ""
    deps.settings.deepseek_api_key = ""
    deps.settings.research_provider = "anthropic"
    deps.settings.research_model = ""
    yield
    # Restored after the test, not before the next: a later file's
    # module-scoped client is built before any per-test setup runs, and must
    # not inherit a test's deliberately broken database.
    deps.settings.database_url = os.environ["DATABASE_URL"]


@pytest.fixture(autouse=True)
def _close_stray_stores(monkeypatch):
    """Close every RunStore a test created and abandoned — a helper's store,
    or the one replaced to simulate a restart. Left open, each leaks its
    SQLite connections until garbage collection, with a ResourceWarning.
    The store currently installed in deps is closed by the next reset."""
    from api import deps, store
    created = []
    original = store.RunStore.__init__

    def tracking(self, *args, **kwargs):
        original(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(store.RunStore, "__init__", tracking)
    yield
    for s in created:
        if s is not deps.runs:
            s.close()


@pytest.fixture(autouse=True, scope="session")
def _close_the_last_store():
    yield
    from api import deps
    deps.jobs.shutdown(wait=True)
    deps.research_jobs.shutdown(wait=True)
    deps.runs.close()
