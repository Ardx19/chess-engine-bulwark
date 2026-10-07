"""Export a trained net: quantised `.nnue`, a float32 `.nnuef` (for the engine's float-vs-PyTorch parity check) and
`ref-<sha8>.jsonl` (float outputs on FENs, for the engine's `verifynet`).

Quantisation: first layer int16 with QA = 255 (weights and biases), output layer int16 weights with QB = 64,
int32 bias scaled by QA*QA*QB. Inference: out = (sum_i clamp(acc_i, 0, QA)^2 * w_q[i] + bias_q) / (QA*QA*QB).
File layout (little endian): magic "NNUEWDL1" | u32 version=1 | u32 arch id (1 N1, 2 N1-WDL) | u32 L1 | u32 buckets=1 |
u32 heads | i16 ft_w[768][L1] | i16 ft_b[L1] | i16 out_w[heads][2*L1] | i32 out_b[heads] | 32-byte sha256 of everything before.
The float file uses magic "NNUEFLT1" with the same layout in float32.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct

import numpy as np
import torch

from pipeline.records import TRAIN_DTYPE, read_shard, to_fen
from .data import PAD, features
from .model import NNUE

QA, QB = 255, 64


def _weights(model: NNUE):
    ft_w = model.ft.weight.detach().cpu().numpy()[:PAD].astype(np.float32)          # (768, N)
    ft_b = model.ft_bias.detach().cpu().numpy().astype(np.float32)
    out_w = model.out.weight.detach().cpu().numpy().astype(np.float32)               # (heads, 2N)
    out_b = model.out.bias.detach().cpu().numpy().astype(np.float32)
    return ft_w, ft_b, out_w, out_b


def export_net(model: NNUE, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ft_w, ft_b, out_w, out_b = _weights(model)
    arch = 1 if model.head == 1 else 2
    hdr = lambda magic: magic + struct.pack("<5I", 1, arch, model.n, 1, model.head)   # noqa: E731
    q_ft_w = np.clip(np.round(ft_w * QA), -32768, 32767).astype("<i2")
    q_ft_b = np.clip(np.round(ft_b * QA), -32768, 32767).astype("<i2")
    q_out_w = np.clip(np.round(out_w * QB), -32768, 32767).astype("<i2")
    q_out_b = np.round(out_b * QA * QA * QB).astype("<i4")
    body = hdr(b"NNUEWDL1") + q_ft_w.tobytes() + q_ft_b.tobytes() + q_out_w.tobytes() + q_out_b.tobytes()
    body += hashlib.sha256(body).digest()
    sha8 = hashlib.sha256(body).hexdigest()[:8]
    qpath = os.path.join(out_dir, f"net-{sha8}.nnue")
    with open(qpath, "wb") as f:
        f.write(body)
    fbody = hdr(b"NNUEFLT1") + ft_w.astype("<f4").tobytes() + ft_b.astype("<f4").tobytes() + out_w.astype("<f4").tobytes() + out_b.astype("<f4").tobytes()
    fbody += hashlib.sha256(fbody).digest()
    fpath = os.path.join(out_dir, f"net-{sha8}.nnuef")
    with open(fpath, "wb") as f:
        f.write(fbody)
    return {"sha8": sha8, "quantized": qpath, "float": fpath}


@torch.no_grad()
def write_reference(model: NNUE, shard_path: str, out_path: str, n: int = 10000):
    """Float model outputs (logits) on the first n positions of a training/master shard, one JSON line each."""
    rec = read_shard(shard_path)[:n]
    model.eval()
    us, them = features(rec)
    out = model(torch.from_numpy(us), torch.from_numpy(them)).float().numpy().reshape(len(rec), -1)
    with open(out_path, "w") as f:
        for i in range(len(rec)):
            f.write(json.dumps({"fen": to_fen(rec[i]), "out": [float(x) for x in out[i]]}) + "\n")


def load_checkpoint_model(path: str) -> NNUE:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or {}
    state = ck["model"]
    n = state["ft_bias"].shape[0]
    head = state["out.weight"].shape[0]
    m = NNUE(n, head)
    m.load_state_dict(state)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--ref-shard")
    ap.add_argument("--ref-n", type=int, default=10000)
    a = ap.parse_args()
    m = load_checkpoint_model(a.ckpt)
    info = export_net(m, a.out_dir)
    if a.ref_shard:
        write_reference(m, a.ref_shard, os.path.join(a.out_dir, f"ref-{info['sha8']}.jsonl"), a.ref_n)
    print(json.dumps(info))
