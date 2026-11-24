"""Build the tablebase-defense suite (B.6): 6-7 piece positions that are draws despite a material deficit, verified
with the Lichess tablebase API (7-piece coverage; cached and throttled). The engine defends the side with less
material; both engines load only 3-5 piece tables, so the oracle that certified the start never adjudicates.

Candidates come from master shards (positions with 6-7 pieces, no castling) and/or EPD files. Only the "draw"
category is kept (not cursed wins / blessed losses, which depend on the halfmove clock), every position starts with
a zero clock, and duplicates are removed. Positions are split into dev / test halves by cluster (pawn structure),
like the other benchmark sets.

Usage:
  python3 scripts/tb_suite.py --shards 'data/master/*.bin' --n 150 --out benchsets/tb_suite
  python3 scripts/tb_suite.py --epd candidates.epd --n 150 --out benchsets/tb_suite
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys

import chess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline.records import read_shard, to_fen          # noqa: E402
from pipeline.tb_check import LichessOracle              # noqa: E402

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(b: chess.Board, color) -> int:
    return sum(v * len(b.pieces(pt, color)) for pt, v in VALUE.items())


def candidates_from_shards(pattern, per_shard=20000):
    import numpy as np
    for p in sorted(glob.glob(pattern)):
        rec = read_shard(p, mmap=True)
        occ = np.asarray(rec["occ"])
        cnt = np.array([bin(int(x)).count("1") for x in occ[:per_shard * 20]])
        cas = (np.asarray(rec["stm_cas"][:len(cnt)]) >> 1) & 15
        idx = np.nonzero((cnt >= 6) & (cnt <= 7) & (cas == 0))[0][:per_shard]
        for i in idx:
            yield to_fen(rec[int(i)])


def candidates_from_epd(path):
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                b, _ = chess.Board.from_epd(line.strip())
                yield b.fen()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards")
    ap.add_argument("--epd", action="append", default=[])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-deficit", type=int, default=1, help="material deficit of the defending side (pawn units)")
    ap.add_argument("--max-queries", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", default="results/tb_api_cache.json")
    ap.add_argument("--out", default="benchsets/tb_suite")
    a = ap.parse_args()

    pool, seen = [], set()
    srcs = ([candidates_from_shards(a.shards)] if a.shards else []) + [candidates_from_epd(p) for p in a.epd]
    for src in srcs:
        for fen in src:
            b = chess.Board(fen)
            b.halfmove_clock = 0
            if not (6 <= chess.popcount(b.occupied) <= 7) or b.castling_rights or b.is_game_over() or b.is_check():
                continue
            dw, db = material(b, chess.WHITE), material(b, chess.BLACK)
            if abs(dw - db) < a.min_deficit:
                continue
            key = b.epd()
            if key not in seen:
                seen.add(key)
                pool.append(b)
    random.Random(a.seed).shuffle(pool)
    orc = LichessOracle(a.cache)
    kept, queried = [], 0
    for b in pool:
        if len(kept) >= a.n or queried >= a.max_queries:
            break
        queried += 1
        if orc.category(b.fen()) == "draw":
            kept.append(b)
    os.makedirs(a.out, exist_ok=True)
    halves = {"dev": [], "test": []}
    for b in kept:
        wp, bp = int(b.pieces_mask(chess.PAWN, chess.WHITE)), int(b.pieces_mask(chess.PAWN, chess.BLACK))
        cl = hashlib.sha1(f"{wp:016x}{bp:016x}{len(b.piece_map())}".encode()).hexdigest()[:8]
        half = "dev" if int(cl, 16) % 2 == 0 else "test"
        worse = "w" if material(b, chess.WHITE) < material(b, chess.BLACK) else "b"
        halves[half].append((b, worse, cl))
    for name, rows in halves.items():
        with open(os.path.join(a.out, f"{name}.epd"), "w") as f:
            for i, (b, worse, cl) in enumerate(rows):
                f.write(f'{b.epd()} id "tb-{name}-{i:04d}"; c0 "worse={worse} cluster={cl} oracle=lichess-draw";\n')
    print(json.dumps({"candidates": len(pool), "queried": queried, "kept": len(kept),
                      "dev": len(halves["dev"]), "test": len(halves["test"])}))


if __name__ == "__main__":
    main()
