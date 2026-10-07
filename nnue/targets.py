"""Label blending: t = 0.6*WDL_lc0 + 0.25*WDL_sf + 0.15*onehot(result), renormalised over the labels present.

All W/D/L triples are [W, D, L] from the side to move's view. A tablebase-overridden record (flags bit 0) uses
the exact outcome in `result` as a one-hot target and ignores the teachers.
"""
from __future__ import annotations

import numpy as np

from pipeline.records import LC0_D_ABSENT, LC0_Q_ABSENT, RESULT_UNKNOWN, SF_WD_ABSENT

BLEND = (0.6, 0.25, 0.15)   # lc0, stockfish, game result (locked)


def lc0_wdl(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Lc0 root q in [-1, 1] and draw probability d -> [W, D, L]: W = (1 + q - d)/2, D = d, L = (1 - q - d)/2."""
    w = (1.0 + q - d) / 2.0
    l = (1.0 - q - d) / 2.0
    out = np.stack([w, d, l], axis=-1).clip(0.0, 1.0)
    return out / out.sum(axis=-1, keepdims=True).clip(1e-9)


def onehot_result(result: np.ndarray) -> np.ndarray:
    """result 2 = win, 1 = draw, 0 = loss (side to move's view) -> [W, D, L] one-hot (zeros if unknown)."""
    out = np.zeros(result.shape + (3,), dtype=np.float32)
    out[result == 2, 0] = 1
    out[result == 1, 1] = 1
    out[result == 0, 2] = 1
    return out


def blend_targets(rec: np.ndarray, weights=BLEND) -> np.ndarray:
    """(B, 3) float32 targets for a batch of master or training records."""
    w_lc, w_sf, w_res = weights
    B = len(rec)
    has_lc = (rec["lc_q"] != LC0_Q_ABSENT) & (rec["lc_d"] != LC0_D_ABSENT)
    has_sf = rec["sf_w"] != SF_WD_ABSENT
    has_res = rec["result"] != RESULT_UNKNOWN
    lc = lc0_wdl(rec["lc_q"].astype(np.float32) / 1e4, rec["lc_d"].astype(np.float32) / 1e4)
    sw = rec["sf_w"].astype(np.float32) / 1000.0
    sd = rec["sf_d"].astype(np.float32) / 1000.0
    sf = np.stack([sw, sd, (1.0 - sw - sd).clip(0.0, 1.0)], axis=-1)
    res = onehot_result(rec["result"])
    wl = np.where(has_lc, w_lc, 0.0).astype(np.float32)
    ws = np.where(has_sf, w_sf, 0.0).astype(np.float32)
    wr = np.where(has_res, w_res, 0.0).astype(np.float32)
    tot = (wl + ws + wr).clip(1e-9)
    t = (wl[:, None] * lc + ws[:, None] * np.where(has_sf[:, None], sf, 0) + wr[:, None] * res) / tot[:, None]
    tb = (rec["flags"] & 1).astype(bool)
    if tb.any():
        t[tb] = res[tb]
    return t.astype(np.float32)


def expected_score(t: np.ndarray) -> np.ndarray:
    return t[..., 0] + 0.5 * t[..., 1]
