"""Merge the training shards of the Kaggle data sessions into one data/training/ folder (train/ val/ test/).

Accepts Kaggle output downloads as zips or as unpacked folders, anywhere under the given paths; only
bulwark_data/s<N>/training/{train,val,test}/*.bin is taken (labeled/master pieces are ignored). Then checks: every
session present once, record counts per split, and exact-board overlap of val/test with train (pipeline.check_splits).

Usage: python3 scripts/merge_kaggle_data.py <downloads dir or zips...> [--out data/training] [--sessions 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline.check_splits import check_training      # noqa: E402
from pipeline.records import read_shard               # noqa: E402

PAT = re.compile(r"(?:^|/)bulwark_data/s(\d+)/training/(train|val|test)/(s\d+-\d+\.bin)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="data/training")
    ap.add_argument("--sessions", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    found = {}                                   # (session, split, name) -> source
    for inp in a.inputs:
        paths = [inp] if os.path.isfile(inp) else glob.glob(os.path.join(inp, "**", "*"), recursive=True)
        for p in paths:
            if p.endswith(".zip") and os.path.isfile(p):
                with zipfile.ZipFile(p) as z:
                    for m in z.namelist():
                        r = PAT.search(m)
                        if r:
                            found.setdefault((int(r[1]), r[2], r[3]), ("zip", p, m))
            elif p.endswith(".bin"):
                r = PAT.search(p.replace(os.sep, "/"))
                if r:
                    found.setdefault((int(r[1]), r[2], r[3]), ("file", p, None))
    sessions = sorted({k[0] for k in found})
    missing = [s for s in a.sessions if s not in sessions]
    print("sessions found:", sessions, "missing:", missing)
    if missing:
        sys.exit("some sessions are missing; download their outputs first")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
    for (s, split, name), (kind, src, member) in sorted(found.items()):
        if s not in a.sessions:
            continue
        dst = os.path.join(a.out, split, name)
        if kind == "zip":
            with zipfile.ZipFile(src) as z, z.open(member) as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
        else:
            shutil.copyfile(src, dst)

    counts = {}
    for split in ("train", "val", "test"):
        for f in sorted(glob.glob(os.path.join(a.out, split, "*.bin"))):
            s = os.path.basename(f).split("-")[0]
            n = len(read_shard(f, mmap=True))
            counts.setdefault(split, {}).setdefault(s, 0)
            counts[split][s] += n
    totals = {split: sum(v.values()) for split, v in counts.items()}
    print(json.dumps({"per_session": counts, "totals": totals}, indent=1))
    print("checking val/test overlap with train ...")
    rep = check_training(a.out, max_overlap=0.05)
    print(json.dumps(rep, indent=1))
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(a.out, "*", "*.bin")))
    print(f"{a.out}: {size / 1e9:.2f} GB, {'OK' if rep['ok'] else 'OVERLAP ABOVE 5%: check before training'}")


if __name__ == "__main__":
    main()
