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
