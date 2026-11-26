"""Teacher agreement on held-out data (B.1 item 5a): log loss and Brier score of trained nets against the blended
teacher targets of the test split (split by game id, never trained or tuned on).

Every net is scored on the same records. For comparability between heads, `escore_*` metrics use only the expected
score E = W + D/2 (binary cross-entropy and squared error against E_t), which a scalar net also predicts; WDL nets
additionally get 3-class log loss and Brier against the soft targets.

Usage: python3 -m nnue.evaluate 'data/training/test/*.bin' ckpt/run-n1/final.pt ckpt/run-n1wdl/final.pt [--max 2000000]
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F

from .data import ShardStream
from .export import load_checkpoint_model
from .model import NNUE, run_model


@torch.no_grad()
def score(models: dict, paths, batch=16384, max_records=0, device="cpu"):
    acc = {k: {"n": 0, "escore_bce": 0.0, "escore_brier": 0.0, "wdl_logloss": 0.0, "wdl_brier": 0.0} for k in models}
    for k, m in models.items():       # same records, same order for every net (one stream per net: inputs differ)
        arch = "n1" if isinstance(m, NNUE) else "n2"
        seen = 0
        for b in ShardStream(paths, batch, seed=0, split=None, prefetch=0, arch=arch):
            t = torch.from_numpy(b["target"]).to(device)
            e = torch.from_numpy(b["escore"]).to(device)
            out = run_model(m, b, device)
            if m.head == 1:
                p = torch.sigmoid(out)
            else:
                # dual head: W/D/L metrics from out[:, 1:4]; the expected-score metrics use the scalar head
                logp = F.log_softmax(out[:, 1:4] if m.head == 4 else out, dim=1)
                pr = logp.exp()
                p = pr[:, 0] + 0.5 * pr[:, 1]
                acc[k]["wdl_logloss"] += float(-(t * logp).sum(1).sum())
                acc[k]["wdl_brier"] += float(((pr - t) ** 2).sum(1).sum())
                if m.head == 4:
                    p = torch.sigmoid(out[:, 0])
            p = p.clamp(1e-6, 1 - 1e-6)
            acc[k]["escore_bce"] += float(-(e * p.log() + (1 - e) * (1 - p).log()).sum())
            acc[k]["escore_brier"] += float(((p - e) ** 2).sum())
            acc[k]["n"] += len(e)
            seen += len(e)
            if b["cursor"]["epoch"] > 0 or (max_records and seen >= max_records):
                break                               # one pass over the shards (remainders < batch are dropped)
    out = {}
    for k, a in acc.items():
        n = max(a["n"], 1)
        out[k] = {"records": a["n"], "escore_bce": a["escore_bce"] / n, "escore_brier": a["escore_brier"] / n}
        if models[k].head in (3, 4):
            out[k].update(wdl_logloss=a["wdl_logloss"] / n, wdl_brier=a["wdl_brier"] / n)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shards")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16384)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for c in a.ckpts:
        m = load_checkpoint_model(c).to(dev)
        m.eval()
        models[c] = m
    print(json.dumps(score(models, sorted(glob.glob(a.shards)), a.batch, a.max, dev), indent=1))
