"""Master / training record formats (frozen in docs/interfaces.md) and game-id splits.

Master record: 56 bytes. Training record: 40 bytes (provenance dropped). All little endian.
Board part (27 bytes): occupancy bitboard (LERF, a1 = bit 0), 16 bytes of 4-bit piece codes in ascending
square order (code = Piece 0..11: white P N B R Q K = 0..5, black = 6..11; low nibble first),
stm/castling byte, ep byte (file + 1, 0 = none), rule50 byte.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

SF_CP_ABSENT = -32768
SF_WD_ABSENT = 0xFFFF
LC0_Q_ABSENT = -32768
LC0_D_ABSENT = 0xFFFF
OWN_CP_ABSENT = -32768
RESULT_UNKNOWN = 255
FLAG_TB = 1  # bit 0: tablebase-overridden; `result` then holds the exact outcome. Bits 1-7: source id.

MASTER_DTYPE = np.dtype([
    ("occ", "<u8"),
    ("pcs", "u1", (16,)),
    ("stm_cas", "u1"),      # bit 0: side to move (1 = black); bits 1-4: castling WK WQ BK BQ
    ("ep", "u1"),
    ("r50", "u1"),
    ("sf_cp", "<i2"),
    ("sf_w", "<u2"),        # per mille
    ("sf_d", "<u2"),
    ("lc_q", "<i2"),        # root_q * 10000
    ("lc_d", "<u2"),        # root_d * 10000
    ("own_cp", "<i2"),
    ("result", "u1"),       # 0 loss, 1 draw, 2 win (side to move's view), 255 unknown
    ("flags", "u1"),
    ("ply", "<u2"),
    ("gid", "<u8"),
    ("rsv", "u1", (5,)),
])
assert MASTER_DTYPE.itemsize == 56

TRAIN_DTYPE = np.dtype([
    ("occ", "<u8"),
    ("pcs", "u1", (16,)),
    ("stm_cas", "u1"),
    ("ep", "u1"),
    ("r50", "u1"),
    ("sf_cp", "<i2"),
    ("sf_w", "<u2"),
    ("sf_d", "<u2"),
    ("lc_q", "<i2"),
    ("lc_d", "<u2"),
    ("result", "u1"),
    ("flags", "u1"),
    ("pad", "u1"),
])
assert TRAIN_DTYPE.itemsize == 40

MASTER_MAGIC = b"NNUEMR01"
TRAIN_MAGIC = b"NNUETR01"
HEADER_FMT = "<8sIIQII"   # magic, version, record_size, count, shard_id, reserved
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 32
VERSION = 1


def write_shard(path: str, records: np.ndarray, shard_id: int = 0) -> None:
    magic = MASTER_MAGIC if records.dtype == MASTER_DTYPE else TRAIN_MAGIC
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, magic, VERSION, records.dtype.itemsize, len(records), shard_id, 0))
        f.write(np.ascontiguousarray(records).tobytes())


def read_shard(path: str, mmap: bool = False) -> np.ndarray:
    with open(path, "rb") as f:
        magic, version, rsize, count, _shard, _ = struct.unpack(HEADER_FMT, f.read(HEADER_SIZE))
    if version != VERSION:
        raise ValueError(f"{path}: unsupported shard version {version}")
    dtype = {MASTER_MAGIC: MASTER_DTYPE, TRAIN_MAGIC: TRAIN_DTYPE}.get(magic)
    if dtype is None or dtype.itemsize != rsize:
        raise ValueError(f"{path}: bad magic/record size ({magic!r}, {rsize})")
    if mmap:
        return np.memmap(path, dtype=dtype, mode="r", offset=HEADER_SIZE, shape=(count,))
    return np.fromfile(path, dtype=dtype, count=count, offset=HEADER_SIZE)


# ---- game-id splits: hash the game id so positions of one game never cross splits ----

def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


TRAIN, VAL, TEST = 0, 1, 2


def split_of(gid: np.ndarray, val_pct: int = 2, test_pct: int = 2) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test, by hash of the game id (deterministic)."""
    with np.errstate(over="ignore"):
        h = (_splitmix64(np.asarray(gid, dtype=np.uint64)) % np.uint64(100)).astype(np.int64)
    out = np.zeros(h.shape, dtype=np.int8)
    out[h < val_pct + test_pct] = VAL
    out[h < test_pct] = TEST
    return out


def to_training(master: np.ndarray) -> np.ndarray:
    t = np.zeros(len(master), dtype=TRAIN_DTYPE)
    for name in TRAIN_DTYPE.names:
        if name != "pad":
            t[name] = master[name]
    return t


# ---- board decode (numpy reference implementation, used by the loader and the tests) ----

def decode_board(rec: np.ndarray):
    """rec: structured array (any record dtype) of length B. Returns (rows, cols, codes) where for every
    occupied square: row = record index, col = square, code = Piece 0..11."""
    B = len(rec)
    occ_bytes = np.ascontiguousarray(rec["occ"]).view(np.uint8).reshape(B, 8)
    bits = np.unpackbits(occ_bytes, axis=1, bitorder="little")           # (B, 64) bit i = square i
    rows, cols = np.nonzero(bits)
    counts = bits.sum(axis=1).astype(np.int64)
    starts = np.cumsum(counts) - counts
    k = np.arange(len(rows)) - starts[rows]                              # index of the piece within its record
    pcs = rec["pcs"]                                                     # (B, 16)
    nib = np.empty((B, 32), dtype=np.uint8)
    nib[:, 0::2] = pcs & 0x0F
    nib[:, 1::2] = pcs >> 4
    return rows, cols, nib[rows, k]


PIECE_CHARS = "PNBRQKpnbrqk"


def to_fen(rec) -> str:
    """FEN for a single record (numpy void scalar or 1-element structured array)."""
    r = np.asarray(rec).reshape(1)
    rows, cols, codes = decode_board(r)
    board = [None] * 64
    for s, c in zip(cols, codes):
        board[int(s)] = PIECE_CHARS[int(c)]
    ranks = []
    for rank in range(7, -1, -1):
        out, empty = "", 0
        for f in range(8):
            p = board[rank * 8 + f]
            if p is None:
                empty += 1
            else:
                out += (str(empty) if empty else "") + p
                empty = 0
        ranks.append(out + (str(empty) if empty else ""))
    sc = int(r["stm_cas"][0])
    cas = "".join(ch for bit, ch in zip((2, 4, 8, 16), "KQkq") if sc & bit) or "-"
    ep = int(r["ep"][0])
    ep_s = "-" if ep == 0 else "abcdefgh"[ep - 1] + ("6" if (sc & 1) == 0 else "3")
    return f"{'/'.join(ranks)} {'b' if sc & 1 else 'w'} {cas} {ep_s} {int(r['r50'][0])} 1"


@dataclass
class ShardInfo:
    name: str
    records: int
    sha256: str


def from_fen(fen: str, **fields) -> np.ndarray:
    """Pack a FEN into a 1-element master record (tests, reference sets). Extra keyword fields are assigned."""
    parts = fen.split()
    board = {}
    rank, f = 7, 0
    for ch in parts[0]:
        if ch == "/":
            rank, f = rank - 1, 0
        elif ch.isdigit():
            f += int(ch)
        else:
            board[rank * 8 + f] = PIECE_CHARS.index(ch)
            f += 1
    rec = np.zeros(1, dtype=MASTER_DTYPE)
    occ = 0
    for k, s in enumerate(sorted(board)):
        occ |= 1 << s
        rec["pcs"][0, k >> 1] |= board[s] << (4 * (k & 1))
    rec["occ"] = occ
    cas = parts[2] if len(parts) > 2 else "-"
    bits = sum(b for ch, b in zip("KQkq", (2, 4, 8, 16)) if ch in cas)
    rec["stm_cas"] = (1 if parts[1] == "b" else 0) | bits
    rec["r50"] = int(parts[4]) if len(parts) > 4 else 0
    rec["sf_cp"] = SF_CP_ABSENT; rec["sf_w"] = SF_WD_ABSENT; rec["sf_d"] = SF_WD_ABSENT
    rec["lc_q"] = LC0_Q_ABSENT; rec["lc_d"] = LC0_D_ABSENT; rec["own_cp"] = OWN_CP_ABSENT
    rec["result"] = RESULT_UNKNOWN
    for k, v in fields.items():
        rec[k] = v
    return rec
