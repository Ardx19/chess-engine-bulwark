"""Carve the frozen defense-benchmark sets out of an unbalanced opening book (B.6, sprint sizes).

Input: UHO_Lichess_4852_v1.epd (or any EPD). For every position Stockfish decides the worse side (fixed nodes,
cached). Positions are grouped into clusters by pawn structure, a proxy for "same source game or opening line"
(UHO lines carry no game ids); a cluster never straddles two sets. Clusters are shuffled with a fixed seed and dealt
into calibration / dev / test / sealed; whatever is left becomes `pool.epd` (SPRT and net-check openings).

Output (default benchsets/): <set>.epd lines like
    <fen> id "dev-0007"; c0 "worse=b sf=-87 cluster=9f3a1c2e";
plus manifest.json (source hash, Stockfish version/hash/nodes, seed, sizes) and SHA256SUMS. Commit the directory
once and never regenerate it after any defense tuning has started.

Usage:
  python3 scripts/make_benchsets.py ~/data/books/UHO_Lichess_4852_v1.epd --sf ~/tools/stockfish/.../stockfish
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

SIZES = (("calibration", 200), ("dev", 300), ("test", 500), ("sealed", 500))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cluster_key(b: chess.Board) -> str:
    wp, bp = int(b.pieces_mask(chess.PAWN, chess.WHITE)), int(b.pieces_mask(chess.PAWN, chess.BLACK))
    return hashlib.sha1(f"{wp:016x}{bp:016x}".encode()).hexdigest()[:8]


def read_epd(path: str):
    out, seen = [], set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            b, _ = chess.Board.from_epd(line)
            key = b.epd()
            if key in seen:
                continue
            seen.add(key)
            out.append(b)
    return out


def sf_evals(boards, sf_path, nodes, workers, cache_path):
    cache = json.load(open(cache_path)) if cache_path and os.path.exists(cache_path) else {}
    todo = [b for b in boards if b.epd() not in cache]

    def run(chunk):
        eng = chess.engine.SimpleEngine.popen_uci(sf_path)
        eng.configure({"Threads": 1, "Hash": 64})
        res = {}
        try:
            for b in chunk:
                info = eng.analyse(b, chess.engine.Limit(nodes=nodes))
                res[b.epd()] = info["score"].white().score(mate_score=32000)
        finally:
            eng.quit()
        return res

    chunks = [todo[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(run, chunks):
            cache.update(r)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump(cache, open(cache_path, "w"))
    name = ""
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    name = eng.id.get("name", "")
    eng.quit()
    return cache, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epd")
    ap.add_argument("--sf", required=True)
    ap.add_argument("--nodes", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-abs", type=int, default=50, help="skip positions with |SF eval| below this (no clear worse side)")
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out", default="benchsets")
    ap.add_argument("--cache", default="results/benchsets_sf_cache.json")
    a = ap.parse_args()

    boards = read_epd(a.epd)
    evals, sf_name = sf_evals(boards, a.sf, a.nodes, a.workers, a.cache)
    clusters: dict[str, list] = {}
    skipped = 0
    for b in boards:
        cp = evals[b.epd()]
        if abs(cp) < a.min_abs:
            skipped += 1
            continue
        clusters.setdefault(cluster_key(b), []).append((b, cp))
    import random
    keys = sorted(clusters)
    random.Random(a.seed).shuffle(keys)
    sets = {name: [] for name, _ in SIZES}
    sets["pool"] = []
    it = iter(keys)
    for name, size in SIZES:
        while len(sets[name]) < size:
            k = next(it, None)
            if k is None:
                break
            sets[name] += [(b, cp, k) for b, cp in clusters[k]]
    for k in it:
        sets["pool"] += [(b, cp, k) for b, cp in clusters[k]]

    os.makedirs(a.out, exist_ok=True)
    sums = []
    for name, rows in sets.items():
        path = os.path.join(a.out, f"{name}.epd")
        with open(path, "w") as f:
            for i, (b, cp, k) in enumerate(rows):
                worse = "w" if cp < 0 else "b"
                f.write(f'{b.epd()} id "{name}-{i:04d}"; c0 "worse={worse} sf={cp} cluster={k}";\n')
        sums.append(f"{sha256(path)}  {name}.epd")
    manifest = {"source": os.path.basename(a.epd), "source_sha256": sha256(a.epd), "positions": len(boards),
                "skipped_small_eval": skipped, "clusters": len(clusters), "seed": a.seed,
                "stockfish": {"name": sf_name, "sha256": sha256(a.sf), "nodes": a.nodes},
                "sizes": {k: len(v) for k, v in sets.items()}, "cluster_key": "sha1(white pawns, black pawns)[:8]"}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    open(os.path.join(a.out, "SHA256SUMS"), "w").write("\n".join(sums) + "\n")
    print(json.dumps(manifest["sizes"]))


if __name__ == "__main__":
    main()
