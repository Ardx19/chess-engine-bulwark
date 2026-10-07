"""Stream an Lc0 training-data tar (or a directory of training.*.gz files) through tools/lc0_convert.

Each gz member is one game; it is decompressed here and sent to the converter as one frame, so the converter
knows the game boundaries (needed for game-id splits). Nothing is kept on disk except the output shard.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import subprocess
import sys
import tarfile
import time

RECORD = 8356
FRAME_MAGIC = 0x47414D45


def iter_games_tar(path: str):
    with tarfile.open(path, "r|*") as tf:      # streaming mode: sequential read, no seeking
        for m in tf:
            if m.isfile() and os.path.basename(m.name).startswith("training.") and m.name.endswith(".gz"):
                with tf.extractfile(m) as f:
                    data = f.read()
                yield m.name, data


def iter_games_dir(path: str):
    for name in sorted(os.listdir(path)):
        if name.startswith("training.") and name.endswith(".gz"):
            with open(os.path.join(path, name), "rb") as f:
                yield name, f.read()


def convert(src: str, out: str, converter: str, session: int, shard_id: int, source_id: int = 0, stride: int = 1,
            syzygy: str | None = None, max_games: int = 0) -> dict:
    cmd = [converter, "--out", out, "--session", str(session), "--shard-id", str(shard_id),
           "--source-id", str(source_id), "--stride", str(stride)]
    if syzygy:
        cmd += ["--syzygy", syzygy]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    games = 0
    members = []
    it = iter_games_tar(src) if os.path.isfile(src) else iter_games_dir(src)
    for name, gz in it:
        try:
            raw = gzip.decompress(gz)
        except (OSError, EOFError):
            continue                              # truncated member: skip the game
        if len(raw) % RECORD:
            continue
        p.stdin.write(struct.pack("<IIII", FRAME_MAGIC, len(raw) // RECORD, games, 0))
        p.stdin.write(raw)
        members.append(name)
        games += 1
        if max_games and games >= max_games:
            break
    p.stdin.close()
    err = p.stderr.read().decode()
    if p.wait() != 0:
        raise RuntimeError(f"converter failed: {err}")
    stats = json.loads(err.strip().splitlines()[-1])
    stats["seconds"] = time.time() - t0
    stats["source"] = os.path.basename(src)
    stats["members"] = len(members)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--converter", default=os.path.join(os.path.dirname(__file__), "..", "build", "lc0_convert"))
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--syzygy")
    ap.add_argument("--max-games", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(convert(a.src, a.out, a.converter, a.session, a.shard_id, 0, a.stride, a.syzygy, a.max_games)))
