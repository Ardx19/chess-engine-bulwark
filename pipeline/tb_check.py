"""Check tablebase-overridden labels against an independent oracle (WP-B data check).

The converter (tools/lc0_convert --syzygy) replaces the target of positions with <= 5 pieces, rule50 == 0 and no
castling rights by the exact WDL outcome, with cursed wins and blessed losses counted as draws (they are draws
under the 50-move rule). This script re-derives that outcome per record from
  * the Lichess tablebase API (`--oracle lichess`, cached and throttled), which reports the cursed-win and
    blessed-loss categories explicitly, or
  * python-chess's Syzygy prober on local tables (`--oracle local --syzygy DIR`), wdl -2..2 with +-1 = cursed/blessed,
and reports agreement per category. Any disagreement is a failure.

Usage:
  python3 -m pipeline.tb_check 'data/master/*.bin' --oracle lichess --sample 300 --cache results/tb_api_cache.json
  python3 -m pipeline.tb_check 'data/master/*.bin' --oracle local --syzygy ~/data/syzygy/tb345
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .records import FLAG_TB, read_shard, to_fen

API = "https://tablebase.lichess.ovh/standard?fen="
# Lichess category (side to move) -> expected `result` byte (0 loss, 1 draw, 2 win) under the 50-move rule.
CATEGORY_RESULT = {"win": 2, "cursed-win": 1, "draw": 1, "blessed-loss": 1, "loss": 0}
# python-chess probe_wdl -> same mapping
WDL_RESULT = {2: 2, 1: 1, 0: 1, -1: 1, -2: 0}
WDL_CATEGORY = {2: "win", 1: "cursed-win", 0: "draw", -1: "blessed-loss", -2: "loss"}


class LichessOracle:
    def __init__(self, cache_path: str | None, delay: float = 1.1):
        self.cache_path = cache_path
        self.delay = delay
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
        self._last = 0.0

    def category(self, fen: str) -> str | None:
        if fen in self.cache:
            return self.cache[fen]
        wait = self.delay - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = API + urllib.parse.quote(fen.replace(" ", "_"), safe="/_-")
        req = urllib.request.Request(url, headers={"User-Agent": "nnue-data-check/1.0"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    cat = json.load(r).get("category")
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:                    # rate limited: back off
                    time.sleep(60 * (attempt + 1))
                    continue
                raise
        else:
            return None
        self._last = time.time()
        self.cache[fen] = cat
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
        return cat


class LocalOracle:
    def __init__(self, path: str):
        import chess.syzygy
        self.tb = chess.syzygy.open_tablebase(os.path.expanduser(path))

    def category(self, fen: str) -> str | None:
        import chess
        try:
            return WDL_CATEGORY[self.tb.probe_wdl(chess.Board(fen))]
        except (KeyError, chess.syzygy.MissingTableError):
            return None


def check(records: np.ndarray, oracle) -> dict:
    by_cat: dict[str, dict] = {}
    mismatches = []
    unknown = 0
    for i in range(len(records)):
        fen = to_fen(records[i])
        cat = oracle.category(fen)
        if cat not in CATEGORY_RESULT:            # unknown / maybe-win / maybe-loss / missing table
            unknown += 1
            continue
        exp, got = CATEGORY_RESULT[cat], int(records["result"][i])
        c = by_cat.setdefault(cat, {"n": 0, "agree": 0})
        c["n"] += 1
        if exp == got:
            c["agree"] += 1
        elif len(mismatches) < 20:
            mismatches.append({"fen": fen, "category": cat, "expected": exp, "record_result": got})
    n = sum(c["n"] for c in by_cat.values())
    agree = sum(c["agree"] for c in by_cat.values())
    return {"checked": n, "agree": agree, "unknown": unknown, "by_category": by_cat, "mismatches": mismatches,
            "ok": n > 0 and agree == n}


def sample_tb_records(paths, sample: int, seed: int = 0) -> np.ndarray:
    parts = []
    for p in sorted(paths):
        rec = read_shard(p, mmap=True)
        parts.append(np.asarray(rec[(np.asarray(rec["flags"]) & FLAG_TB) != 0]))
    rec = np.concatenate(parts) if parts else np.zeros(0)
    if sample and len(rec) > sample:
        rec = rec[np.random.default_rng(seed).choice(len(rec), sample, replace=False)]
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("master_glob")
    ap.add_argument("--oracle", choices=("lichess", "local"), default="lichess")
    ap.add_argument("--syzygy", default="~/data/syzygy/tb345")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--cache", default="results/tb_api_cache.json")
    a = ap.parse_args()
    recs = sample_tb_records(glob.glob(a.master_glob), a.sample)
    orc = LichessOracle(a.cache) if a.oracle == "lichess" else LocalOracle(a.syzygy)
    rep = check(recs, orc)
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["ok"] else 1)
