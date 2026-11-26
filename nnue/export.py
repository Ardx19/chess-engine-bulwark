"""Export a trained net: quantised `.nnue`, a float32 `.nnuef` (for the engine's float-vs-PyTorch parity check) and
`ref-<sha8>.jsonl` (float outputs on FENs, for the engine's `verifynet`).

Quantisation: first layer int16 with QA = 255 (weights and biases), output layer int16 weights with QB = 64,
int32 bias scaled by QA*QA*QB. Inference: out = (sum_i clamp(acc_i, 0, QA)^2 * w_q[i] + bias_q) / (QA*QA*QB).
File layout (little endian): magic "NNUEWDL1" | u32 version=1 | u32 arch id (1 N1, 2 N1-WDL) | u32 L1 | u32 buckets=1 |
u32 heads | i16 ft_w[768][L1] | i16 ft_b[L1] | i16 out_w[heads][2*L1] | i32 out_b[heads] | 32-byte sha256 of everything before.
The float file uses magic "NNUEFLT1" with the same layout in float32.

Version 2 (N2 = arch 3, N3 = arch 4): magic | u32 version=2 | arch | L1 | buckets=8 | heads | L2 | L3 | K | router
(0 none, 1 piece count, 2 learned) | i16 ft_w[8*768][L1] | i16 ft_b[L1] | i8 l2_w[K*L2][2*L1] (QB = 64) | f32 l2_b[K*L2] |
f32 l3_w[K][L3][L2] | f32 l3_b[K][L3] | f32 out_w[K][heads][L3] | f32 out_b[K][heads] |
(router 2: i8 r_w[K][2*L1] (QB) | f32 r_s[K][4] | f32 r_b[K]) | sha256. Float v2: same order, everything f32.
L2 inputs in the engine are x_q = round(clamp(acc, 0, QA)^2 / 512) in 0..127, so l2 = sum(x_q * w_q) / (127 * 64) + b.
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
from .data import PAD, make_batch
from .model import NNUE, NNUE2, NNUE4, run_model

QA, QB = 255, 64


def _weights(model: NNUE):
    ft_w = model.ft.weight.detach().cpu().numpy()[:PAD].astype(np.float32)          # (768, N)
    ft_b = model.ft_bias.detach().cpu().numpy().astype(np.float32)
    out_w = model.out.weight.detach().cpu().numpy().astype(np.float32)               # (heads, 2N)
    out_b = model.out.bias.detach().cpu().numpy().astype(np.float32)
    if model.head == 4:                                                             # dual: [scalar, W, D, L]
        out_w = np.concatenate([model.out_es.weight.detach().cpu().numpy().astype(np.float32), out_w])
        out_b = np.concatenate([model.out_es.bias.detach().cpu().numpy().astype(np.float32), out_b])
    return ft_w, ft_b, out_w, out_b


def _write_pair(out_dir, qbody: bytes, fbody: bytes) -> dict:
    qbody += hashlib.sha256(qbody).digest()
    fbody += hashlib.sha256(fbody).digest()
    sha8 = hashlib.sha256(qbody).hexdigest()[:8]
    qpath, fpath = os.path.join(out_dir, f"net-{sha8}.nnue"), os.path.join(out_dir, f"net-{sha8}.nnuef")
    with open(qpath, "wb") as f:
        f.write(qbody)
    with open(fpath, "wb") as f:
        f.write(fbody)
    return {"sha8": sha8, "quantized": qpath, "float": fpath}


def export_net_v2(model: NNUE2, out_dir: str) -> dict:
    f32 = lambda t: t.detach().cpu().float().numpy().astype("<f4")        # noqa: E731
    arch = 3 if model.k == 1 else 4
    router = {"none": 0, "count": 1, "learned": 2}[model.router]
    hdr = lambda magic: magic + struct.pack("<9I", 2, arch, model.n, 8, model.head, model.l2n, model.l3n, model.k, router)  # noqa: E731
    from .data import N2_PAD
    ft_w = f32(model.ft.weight)[:N2_PAD]
    ft_b, l2_w, l2_b = f32(model.ft_bias), f32(model.l2.weight), f32(model.l2.bias)
    tail = [f32(model.l3_w), f32(model.l3_b), f32(model.out_w), f32(model.out_b)]
    q16 = lambda a, s: np.clip(np.round(a * s), -32768, 32767).astype("<i2")      # noqa: E731
    q8 = lambda a: np.clip(np.round(a * QB), -127, 127).astype("i1")              # noqa: E731
    qbody = hdr(b"NNUEWDL1") + q16(ft_w, QA).tobytes() + q16(ft_b, QA).tobytes() + q8(l2_w).tobytes() + l2_b.tobytes()
    fbody = hdr(b"NNUEFLT1") + ft_w.tobytes() + ft_b.tobytes() + l2_w.tobytes() + l2_b.tobytes()
    for t in tail:
        qbody += t.tobytes(); fbody += t.tobytes()
    if router == 2:
        rw, rs, rb = f32(model.router_x.weight), f32(model.router_s.weight), f32(model.router_x.bias)
        qbody += q8(rw).tobytes() + rs.tobytes() + rb.tobytes()
        fbody += rw.tobytes() + rs.tobytes() + rb.tobytes()
    os.makedirs(out_dir, exist_ok=True)
    return _write_pair(out_dir, qbody, fbody)


def export_net_v3(model: NNUE4, out_dir: str) -> dict:
    """Version 3 (N4, arch 5): magic | u32 version=3 | arch=5 | L1 | buckets=8 | heads | out_buckets |
    i16 ft_w[8*768][L1] | i16 ft_b[L1] | i16 out_w[out_buckets][heads][2*L1] (QB) | i32 out_b[out_buckets][heads]
    (QA*QA*QB) | sha256. Heads are ordered [scalar] + [W, D, L] as present. Float file: same layout, f32."""
    from .data import N2_PAD
    f32 = lambda t: t.detach().cpu().float().numpy()        # noqa: E731
    ft_w, ft_b = f32(model.ft.weight)[:N2_PAD], f32(model.ft_bias)
    ws, bs = [], []
    if model.head in (1, 4):
        ws.append(f32(model.out_es.weight).reshape(model.ob, 1, -1)); bs.append(f32(model.out_es.bias).reshape(model.ob, 1))
    if model.head in (3, 4):
        ws.append(f32(model.out.weight).reshape(model.ob, 3, -1)); bs.append(f32(model.out.bias).reshape(model.ob, 3))
    out_w, out_b = np.concatenate(ws, axis=1), np.concatenate(bs, axis=1)      # (ob, heads, 2N), (ob, heads)
    hdr = lambda magic: magic + struct.pack("<6I", 3, 5, model.n, 8, model.head, model.ob)   # noqa: E731
    q16 = lambda a, s: np.clip(np.round(a * s), -32768, 32767).astype("<i2")                 # noqa: E731
    qbody = (hdr(b"NNUEWDL1") + q16(ft_w, QA).tobytes() + q16(ft_b, QA).tobytes() + q16(out_w, QB).tobytes()
             + np.round(out_b * QA * QA * QB).astype("<i4").tobytes())
    fbody = (hdr(b"NNUEFLT1") + ft_w.astype("<f4").tobytes() + ft_b.astype("<f4").tobytes()
             + out_w.astype("<f4").tobytes() + out_b.astype("<f4").tobytes())
    os.makedirs(out_dir, exist_ok=True)
    return _write_pair(out_dir, qbody, fbody)


def export_net(model, out_dir: str) -> dict:
    if isinstance(model, NNUE4):
        return export_net_v3(model, out_dir)
    if isinstance(model, NNUE2):
        return export_net_v2(model, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ft_w, ft_b, out_w, out_b = _weights(model)
    arch = 1 if model.head == 1 else 2       # dual (4 heads) keeps arch 2; the head count tells them apart
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
    batch = make_batch(rec, arch="n1" if isinstance(model, NNUE) else "n2")
    out = run_model(model, batch, "cpu").float().numpy().reshape(len(rec), -1)
    with open(out_path, "w") as f:
        for i in range(len(rec)):
            f.write(json.dumps({"fen": to_fen(rec[i]), "out": [float(x) for x in out[i]]}) + "\n")


def load_checkpoint_model(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or {}
    state = ck["model"]
    n = state["ft_bias"].shape[0]
    from .data import N2_PAD
    if "l3_w" not in state and state["ft.weight"].shape[0] == N2_PAD + 1:     # N4
        head = (1 if "out_es.weight" in state else 0) + (3 if "out.weight" in state else 0)
        ob = (state["out_es.weight"] if "out_es.weight" in state else state["out.weight"]).shape[0] // (1 if "out_es.weight" in state else 3)
        m = NNUE4(n, head, bool(cfg.get("wdl_detach") or 0), cfg.get("wdl_weight") or 1.0, ob)
        m.load_state_dict(state)
        return m
    if "l3_w" in state:
        k, l3, l2 = state["l3_w"].shape
        router = "learned" if "router_x.weight" in state else ("count" if k > 1 else "none")
        m = NNUE2(n, state["out_w"].shape[1], l2, l3, k, router, cfg.get("balance", 0.01))
        m.load_state_dict(state)
        return m
    head = 4 if "out_es.weight" in state else state["out.weight"].shape[0]
    m = NNUE(n, head, bool(cfg.get("wdl_detach") or 0), cfg.get("wdl_weight") or 1.0)
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
