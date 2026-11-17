"""Append-only audit log (JSON Lines). One record per question answered.

Records who asked, what, which documents were shown, and the outcome, so
every AI response can be traced back. Chunk text is not logged.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **event) -> dict:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        if self.path is not None:
            line = json.dumps(entry, ensure_ascii=False)
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return entry
