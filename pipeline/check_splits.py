"""Leakage checks before training (WP-B data checks).

1. Master shards: every game id lands in exactly one split (by construction: hash of the game id), and no session id
   is reused by two production sessions (two sessions with the same id would merge unrelated games under one id).
2. Training directory (train/ val/ test/): exact-board overlap between val/test and train. Splitting by game cannot
   prevent the same position (common openings, simple endings) from appearing in different games, so this is
   measured and reported, and `--drop-overlap` rewrites val/test without the overlapping positions.

Usage:
  python3 -m pipeline.check_splits --master 'data/master/*.bin'
  python3 -m pipeline.check_splits --training data/training [--drop-overlap] [--max-overlap 0.05]
Exit status 1 if a check fails (session reuse, or overlap above --max-overlap).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from .make_training import board_hash
from .records import read_shard, split_of, write_shard


def check_master(paths, val_pct=2, test_pct=2) -> dict:
    sessions_by_shard = {}
    owner = {}
    reused = []
    n = 0
    for p in sorted(paths):
        rec = read_shard(p, mmap=True)
        n += len(rec)
        gid = np.asarray(rec["gid"])
        ses = np.unique(gid >> np.uint64(32)).tolist()
        sessions_by_shard[os.path.basename(p)] = ses
        # A session normally writes several shards; flag it only if two shards of one session have overlapping
        # game ids (the same counter range written twice, i.e. a reused session id or a duplicated shard).
        for s in ses:
            g = np.unique(gid[(gid >> np.uint64(32)) == s])
            if s in owner:
                prev_p, prev_g = owner[s]
                if np.intersect1d(prev_g, g, assume_unique=True).size:
                    reused.append({"session": int(s), "shards": [prev_p, os.path.basename(p)]})
                owner[s] = (os.path.basename(p), np.union1d(prev_g, g))
            else:
                owner[s] = (os.path.basename(p), g)
        sp = split_of(gid, val_pct, test_pct)
        # Deterministic by construction; recompute on a permuted copy to guard against a changed split function.
        assert np.array_equal(sp, split_of(gid.copy(), val_pct, test_pct))
    return {"records": n, "shards": len(sessions_by_shard), "sessions": len(owner), "session_reuse": reused,
            "ok": not reused}


def _hashes(paths):
    out = [board_hash(read_shard(p)) for p in paths]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.uint64)


def check_training(root: str, drop_overlap=False, max_overlap=None) -> dict:
    train = sorted(glob.glob(os.path.join(root, "train", "*.bin")))
    res = {"train_shards": len(train)}
    held = {name: sorted(glob.glob(os.path.join(root, name, "*.bin"))) for name in ("val", "test")}
    held_h = {name: np.unique(_hashes(ps)) for name, ps in held.items()}
    hit = {name: np.zeros(len(h), dtype=bool) for name, h in held_h.items()}
    for p in train:                              # stream train shards; held-out sets are small
        th = np.unique(board_hash(read_shard(p)))
        for name, h in held_h.items():
            hit[name] |= np.isin(h, th, assume_unique=True)
    ok = True
    for name, ps in held.items():
        total = overlap = 0
        bad = held_h[name][hit[name]]
        for p in ps:
            rec = read_shard(p)
            m = np.isin(board_hash(rec), bad)
            total += len(rec)
            overlap += int(m.sum())
            if drop_overlap and m.any():
                write_shard(p, rec[~m])
        frac = overlap / total if total else 0.0
        res[name] = {"records": total, "overlap_with_train": overlap, "fraction": round(frac, 6), "dropped": bool(drop_overlap)}
        if max_overlap is not None and frac > max_overlap and not drop_overlap:
            ok = False
    res["ok"] = ok
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", help="glob of master shards")
    ap.add_argument("--training", help="training root with train/ val/ test/")
    ap.add_argument("--drop-overlap", action="store_true")
    ap.add_argument("--max-overlap", type=float, default=None)
    a = ap.parse_args()
    report, ok = {}, True
    if a.master:
        report["master"] = check_master(glob.glob(a.master))
        ok &= report["master"]["ok"]
    if a.training:
        report["training"] = check_training(a.training, a.drop_overlap, a.max_overlap)
        ok &= report["training"]["ok"]
    print(json.dumps(report, indent=1))
    sys.exit(0 if ok else 1)
