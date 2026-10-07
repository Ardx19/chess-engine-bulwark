"""manifest.json writer (schema in docs/interfaces.md)."""
from __future__ import annotations

import json
import time

from .relabel import sha256_file


def write_manifest(path: str, *, lc0_run: str, files: list[str], filter_settings: dict, counts: dict,
                   stockfish: dict | None, shards: list[str], extra: dict | None = None) -> dict:
    m = {
        "schema": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lc0_run": lc0_run,
        "files": files,
        "filter": {**filter_settings, "seen": counts.get("seen", 0), "kept": counts.get("kept", 0), "counts": counts},
        "stockfish": stockfish,
        "shards": [{"name": s, "sha256": sha256_file(s)} for s in shards],
        "license": "Lc0 training data is ODbL 1.0 (credit: Leela Chess Zero); derived datasets stay ODbL.",
    }
    if extra:
        m.update(extra)
    with open(path, "w") as f:
        json.dump(m, f, indent=1)
    return m
