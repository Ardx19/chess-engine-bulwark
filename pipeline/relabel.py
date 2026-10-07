"""Stockfish relabeling: fill sf_cp / sf_w / sf_d of master records (one Stockfish process per worker, Threads=1).

Stockfish's reported W/D/L (UCI_ShowWDL) comes from its own version-specific model, so it is a teacher estimate,
not ground truth; the binary hash and node budget go into the manifest.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import re
import subprocess
import time

import numpy as np

from .records import MASTER_DTYPE, SF_CP_ABSENT, SF_WD_ABSENT, read_shard, to_fen, write_shard

_SCORE = re.compile(r"score (cp|mate) (-?\d+)(?: (?:lowerbound|upperbound))?(?: wdl (\d+) (\d+) (\d+))?")


class Stockfish:
    def __init__(self, path: str, hash_mb: int = 16, syzygy: str | None = None):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._send("uci")
        self.name = ""
        for line in self._lines():
            if line.startswith("id name"):
                self.name = line[8:].strip()
            if line == "uciok":
                break
        self._send(f"setoption name Threads value 1\nsetoption name Hash value {hash_mb}\n"
                   "setoption name UCI_ShowWDL value true")
        if syzygy:
            self._send(f"setoption name SyzygyPath value {syzygy}")
        self._send("isready")
        for line in self._lines():
            if line == "readyok":
                break

    def _send(self, s: str):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _lines(self):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("stockfish died")
            yield line.rstrip("\n")

    def analyse(self, fen: str, nodes: int):
        """Returns (cp, w, d) from the side to move's view. Mate scores map to +-(32000 - |n|) cp, W/D/L as reported."""
        self._send(f"position fen {fen}\ngo nodes {nodes}")
        cp = w = d = None
        for line in self._lines():
            if line.startswith("info") and " score " in line and " pv " in line:
                m = _SCORE.search(line)
                if m:
                    kind, val = m.group(1), int(m.group(2))
                    cp = val if kind == "cp" else (32000 - abs(val)) * (1 if val > 0 else -1)
                    if m.group(3) is not None:
                        w, d = int(m.group(3)), int(m.group(4))
            elif line.startswith("bestmove"):
                break
        return cp, w, d

    def close(self):
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


_SF = None


def _work(args):
    path, nodes, syzygy, recs = args
    global _SF
    if _SF is None:
        _SF = Stockfish(path, syzygy=syzygy)
    n = len(recs)
    cp = np.full(n, SF_CP_ABSENT, dtype=np.int16)
    w = np.full(n, SF_WD_ABSENT, dtype=np.uint16)
    d = np.full(n, SF_WD_ABSENT, dtype=np.uint16)
    for i in range(n):
        c, ww, dd = _SF.analyse(to_fen(recs[i]), nodes)
        if c is not None:
            cp[i] = max(-32000, min(32000, c))
        if ww is not None:
            w[i], d[i] = ww, dd
    return cp, w, d


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def relabel(in_shard: str, out_shard: str, sf_path: str, nodes: int = 2000, workers: int | None = None,
            chunk: int = 2000, syzygy: str | None = None, limit: int = 0) -> dict:
    recs = read_shard(in_shard)
    if limit:
        recs = recs[:limit]
    workers = workers or os.cpu_count() or 1
    jobs = [(sf_path, nodes, syzygy, recs[i:i + chunk]) for i in range(0, len(recs), chunk)]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_work, jobs, chunksize=1)
    out = recs.copy()
    pos = 0
    for (cp, w, d), job in zip(results, jobs):
        n = len(job[3])
        out["sf_cp"][pos:pos + n] = cp
        out["sf_w"][pos:pos + n] = w
        out["sf_d"][pos:pos + n] = d
        pos += n
    secs = time.time() - t0
    write_shard(out_shard, out)
    probe = Stockfish(sf_path)
    name = probe.name
    probe.close()
    return {"records": len(out), "nodes": nodes, "workers": workers, "seconds": secs,
            "relabels_per_s": len(out) / max(secs, 1e-9), "relabels_per_s_per_worker": len(out) / max(secs, 1e-9) / workers,
            "stockfish": name, "stockfish_sha256": sha256_file(sf_path), "hash_mb": 16, "threads": 1,
            "tb": bool(syzygy)}
