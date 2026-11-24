"""Training/master shards -> bullet text format, for the N1 cross-check against bullet's `simple` example.

One line per position:  <FEN> | <score> | <result>
  score  = 400 * logit(E_t) in centipawn-like units, from White's view, clamped to +-3000, where E_t = W + D/2 of the
           blended target (the same target N1 trains on; see nnue/targets.py);
  result = game result from White's view: 1.0 / 0.5 / 0.0 (tablebase-overridden records carry the exact outcome).
Records without a known game result are skipped (bullet needs one). Convert the text to bulletformat with
bullet-utils (`convert --from text`) and train the `simple` example with eval scale 400 and WDL weight 0, which
targets sigmoid(score/400) = E_t, i.e. the same objective as nnue/train.py with --head 1.

Usage: python3 -m pipeline.to_bullet 'data/training/train/*.bin' out.txt [--limit N]
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

from nnue.targets import blend_targets, expected_score
from .records import RESULT_UNKNOWN, read_shard, to_fen

SCORE_CLAMP = 3000


def lines(rec: np.ndarray):
    t = blend_targets(rec)
    e = np.clip(expected_score(t), 1e-6, 1 - 1e-6)
    score_stm = np.clip(np.round(400.0 * np.log(e / (1 - e))), -SCORE_CLAMP, SCORE_CLAMP).astype(int)
    black = (rec["stm_cas"] & 1).astype(bool)
    for i in range(len(rec)):
        r = int(rec["result"][i])
        if r == RESULT_UNKNOWN:
            continue
        res_stm = r / 2.0                                   # 0 loss, 0.5 draw, 1 win (side to move)
        score = -score_stm[i] if black[i] else score_stm[i]
        res = 1.0 - res_stm if black[i] else res_stm
        yield f"{to_fen(rec[i])} | {score} | {res:.1f}"


def convert(paths, out_path: str, limit: int = 0) -> dict:
    n = skipped = 0
    with open(out_path, "w") as f:
        for p in sorted(paths):
            rec = read_shard(p)
            for line in lines(rec):
                f.write(line + "\n")
                n += 1
                if limit and n >= limit:
                    return {"written": n}
            skipped += int((rec["result"] == RESULT_UNKNOWN).sum())
    return {"written": n, "skipped_unknown_result": skipped}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_glob")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    print(convert(glob.glob(a.shard_glob), a.out, a.limit))
