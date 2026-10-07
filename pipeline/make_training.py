"""Master shards -> training shards (40 B records), split by game id into train / val / test directories.

Split is by hash of the game id, so no game crosses a split. Exact-duplicate positions (same board + stm +
castling) are optionally dropped *within* each split.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .records import TEST, TRAIN, VAL, read_shard, split_of, to_training, write_shard

NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


def board_hash(rec: np.ndarray) -> np.ndarray:
    h = rec["occ"].astype(np.uint64)
    pc = np.ascontiguousarray(rec["pcs"]).view("<u8").reshape(len(rec), 2)
    with np.errstate(over="ignore"):
        h = h * np.uint64(0x9E3779B97F4A7C15) ^ pc[:, 0] * np.uint64(0xC2B2AE3D27D4EB4F) ^ pc[:, 1] * np.uint64(0x165667B19E3779F9)
        h ^= rec["stm_cas"].astype(np.uint64) * np.uint64(0xFF51AFD7ED558CCD)
    return h


def make_training(master_paths, out_dir, shard_records=5_000_000, dedup=True, val_pct=2, test_pct=2, prefix="s"):
    counts = {TRAIN: 0, VAL: 0, TEST: 0}
    seen = {s: set() for s in counts}
    n_shards = {s: 0 for s in counts}
    for sp in counts:
        os.makedirs(os.path.join(out_dir, NAMES[sp]), exist_ok=True)
    pending = {s: [] for s in counts}
    pend_n = {s: 0 for s in counts}

    def flush(sp, force=False):
        while pend_n[sp] >= shard_records or (force and pend_n[sp] > 0):
            arr = np.concatenate(pending[sp])
            take, rest = arr[:shard_records], arr[shard_records:]
            path = os.path.join(out_dir, NAMES[sp], f"{prefix}-{n_shards[sp]:05d}.bin")
            write_shard(path, take, n_shards[sp])
            n_shards[sp] += 1
            counts[sp] += len(take)
            pending[sp] = [rest] if len(rest) else []
            pend_n[sp] = len(rest)

    for mp_ in master_paths:
        rec = read_shard(mp_)
        split = split_of(rec["gid"], val_pct, test_pct)
        for sp in counts:
            part = rec[split == sp]
            if not len(part):
                continue
            if dedup:
                h = board_hash(part)
                keep = np.zeros(len(part), dtype=bool)
                for i, x in enumerate(h.tolist()):
                    if x not in seen[sp]:
                        seen[sp].add(x)
                        keep[i] = True
                part = part[keep]
            t = to_training(part)
            pending[sp].append(t)
            pend_n[sp] += len(t)
            flush(sp)
    for sp in counts:
        flush(sp, force=True)
    return {NAMES[s]: counts[s] for s in counts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("master_glob")
    ap.add_argument("out_dir")
    ap.add_argument("--shard-records", type=int, default=5_000_000)
    ap.add_argument("--no-dedup", action="store_true")
    a = ap.parse_args()
    print(make_training(sorted(glob.glob(a.master_glob)), a.out_dir, a.shard_records, not a.no_dedup))
