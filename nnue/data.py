"""Training-record loader: deterministic shard streaming with a resumable cursor, vectorised feature decode.

Feature index (must match the C++ engine): for perspective p and a piece of color c, type t on square s:
    rel = 0 if c == p else 1;   sq = s ^ 56 if p == BLACK else s;   feature = (rel * 6 + t) * 64 + sq     (0..767)
`us` = the side to move's perspective, `them` = the other. Padding index is 768 (a zero embedding).
"""
from __future__ import annotations

import queue
import threading

import numpy as np

from pipeline.records import TRAIN, read_shard, decode_board, split_of
from .targets import BLEND, blend_targets, expected_score

PAD = 768
MAX_PIECES = 32

# N2/N3 inputs: horizontally mirrored king buckets. For perspective p with own king on ksq (seen from p, i.e. ^56 for
# black): mirror = file(ksq) >= 4 (then every square is file-flipped, ^7); bucket = N2_BUCKETS[rank * 4 + mirrored file].
# feature = bucket * 768 + (rel * 6 + type) * 64 + (sq_rel ^ (7 if mirror else 0)). Must match src/nnue.h.
N2_BUCKETS = np.array([0, 1, 2, 3,
                       4, 4, 5, 5,
                       6, 6, 6, 6,
                       6, 6, 6, 6,
                       7, 7, 7, 7,
                       7, 7, 7, 7,
                       7, 7, 7, 7,
                       7, 7, 7, 7], dtype=np.int64)
N_BUCKETS = 8
N2_PAD = N_BUCKETS * 768
N_SCALARS = 4
PHASE_W = np.array([0, 1, 1, 2, 4, 0], dtype=np.float32)
MAT_W = np.array([1, 3, 3, 5, 9, 0], dtype=np.float32)


def features(rec: np.ndarray):
    """-> idx_us, idx_them  (B, 32) int64, padded with 768."""
    B = len(rec)
    rows, sqs, codes = decode_board(rec)
    color = codes // 6
    ptype = codes % 6
    stm = (rec["stm_cas"] & 1).astype(np.int64)[rows]          # 0 white, 1 black
    counts = np.bincount(rows, minlength=B)
    starts = np.cumsum(counts) - counts
    k = np.arange(len(rows)) - starts[rows]
    out = []
    for persp in (stm, 1 - stm):
        rel = (color != persp).astype(np.int64)
        sq = np.where(persp == 1, sqs ^ 56, sqs)
        feat = (rel * 6 + ptype) * 64 + sq
        arr = np.full((B, MAX_PIECES), PAD, dtype=np.int64)
        arr[rows, k] = feat
        out.append(arr)
    return out[0], out[1]


def features_n2(rec: np.ndarray):
    """King-bucketed, mirrored features -> idx_us, idx_them (B, 32) int64 padded with N2_PAD; plus router scalars
    (B, 4) float32 [phase, material balance / 10 (stm view), rule50 / 100, stm is black] and piece counts (B,)."""
    B = len(rec)
    rows, sqs, codes = decode_board(rec)
    color = (codes // 6).astype(np.int64)
    ptype = (codes % 6).astype(np.int64)
    stm = (rec["stm_cas"] & 1).astype(np.int64)
    ksq = np.zeros((2, B), dtype=np.int64)
    kings = ptype == 5
    ksq[color[kings], rows[kings]] = sqs[kings]
    counts = np.bincount(rows, minlength=B)
    starts = np.cumsum(counts) - counts
    k = np.arange(len(rows)) - starts[rows]
    out = []
    for persp_rec in (stm, 1 - stm):
        persp = persp_rec[rows]
        own_k = ksq[persp, rows]
        k_rel = np.where(persp == 1, own_k ^ 56, own_k)
        mirror = (k_rel & 7) >= 4
        f = np.where(mirror, 7 - (k_rel & 7), k_rel & 7)
        bucket = N2_BUCKETS[(k_rel >> 3) * 4 + f]
        sq = np.where(persp == 1, sqs ^ 56, sqs) ^ np.where(mirror, 7, 0)
        rel = (color != persp).astype(np.int64)
        feat = bucket * 768 + (rel * 6 + ptype) * 64 + sq
        arr = np.full((B, MAX_PIECES), N2_PAD, dtype=np.int64)
        arr[rows, k] = feat
        out.append(arr)
    phase = np.minimum(np.bincount(rows, weights=PHASE_W[ptype], minlength=B), 24) / 24.0
    sign = np.where(color == stm[rows], 1.0, -1.0)
    bal = np.bincount(rows, weights=MAT_W[ptype] * sign, minlength=B) / 10.0
    sc = np.stack([phase, bal, rec["r50"].astype(np.float64) / 100.0, stm.astype(np.float64)], axis=1).astype(np.float32)
    return out[0], out[1], sc, counts.astype(np.int64)


def make_batch(rec: np.ndarray, weights=BLEND, arch: str = "n1"):
    t = blend_targets(rec, weights)
    b = {"target": t, "escore": expected_score(t).astype(np.float32)}
    if arch == "n1":
        b["us"], b["them"] = features(rec)
    else:
        b["us"], b["them"], b["scalars"], b["pcount"] = features_n2(rec)
    return b


class ShardStream:
    """Iterates batches over training shards. The seed fixes the shard permutation per epoch and the record
    permutation inside each shard, so (epoch, shard position, offset) is a complete data cursor.
    Records are batched within a shard; the < batch_size remainder of each shard is dropped."""

    def __init__(self, paths, batch_size, seed=0, weights=BLEND, cursor=None, split=TRAIN, prefetch=4, arch="n1"):
        self.paths = sorted(paths)
        self.arch = arch
        self.bs = batch_size
        self.seed = seed
        self.weights = weights
        self.split = split
        self.prefetch = prefetch
        self.cursor = dict(cursor) if cursor else {"epoch": 0, "shard": 0, "offset": 0}

    def _shard_order(self, epoch):
        return np.random.default_rng([self.seed, epoch]).permutation(len(self.paths))

    def _records(self, path, epoch, shard_id):
        rec = read_shard(path)
        if self.split is not None and "gid" in rec.dtype.names:
            rec = rec[split_of(rec["gid"]) == self.split]
        perm = np.random.default_rng([self.seed, epoch, shard_id]).permutation(len(rec))
        return rec[perm]

    def _generate(self):
        cur = dict(self.cursor)
        while True:
            order = self._shard_order(cur["epoch"])
            produced = 0
            while cur["shard"] < len(order):
                sid = int(order[cur["shard"]])
                rec = self._records(self.paths[sid], cur["epoch"], sid)
                nb = len(rec) // self.bs
                for b in range(cur["offset"] // self.bs, nb):
                    chunk = rec[b * self.bs:(b + 1) * self.bs]
                    if b + 1 < nb:
                        nxt = {"epoch": cur["epoch"], "shard": cur["shard"], "offset": (b + 1) * self.bs}
                    elif cur["shard"] + 1 < len(order):
                        nxt = {"epoch": cur["epoch"], "shard": cur["shard"] + 1, "offset": 0}
                    else:
                        nxt = {"epoch": cur["epoch"] + 1, "shard": 0, "offset": 0}
                    batch = make_batch(chunk, self.weights, self.arch)
                    batch["cursor"] = nxt          # where the stream resumes after this batch
                    produced += 1
                    yield batch
                cur = {"epoch": cur["epoch"], "shard": cur["shard"] + 1, "offset": 0}
            if produced == 0 and cur["epoch"] == self.cursor["epoch"]:
                raise RuntimeError("no batches: shards smaller than the batch size, or all records in other splits")
            cur = {"epoch": cur["epoch"] + 1, "shard": 0, "offset": 0}

    def __iter__(self):
        if self.prefetch <= 0:
            yield from self._generate()
            return
        q = queue.Queue(self.prefetch)
        stop = threading.Event()

        def run():
            try:
                for b in self._generate():
                    while not stop.is_set():
                        try:
                            q.put(b, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        return
            except Exception as e:      # surface loader errors in the training thread
                q.put(e)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        try:
            while True:
                b = q.get()
                if isinstance(b, Exception):
                    raise b
                yield b
        finally:
            stop.set()
