"""Trainer with the B.9 checkpoint/resume spec.

ckpt/run-<id>/{config.yaml, step-<n>.pt (last 3), best.pt, metrics.jsonl}
Checkpoint = model, optimizer, scheduler, AMP scaler, global step, data cursor, RNG states (python, numpy, torch,
cuda), best validation loss, wall-clock used. Atomic writes (temp file, fsync, os.replace). `--deadline 11h30m`
and SIGTERM both produce a final checkpoint and export. `--resume auto` searches the local ckpt dir, then
/kaggle/input/*/ckpt, then the Hugging Face model repo in $HF_REPO.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import signal
import time

import numpy as np
import torch

from pipeline.records import VAL
from .data import ShardStream
from .model import NNUE2, build_model, loss_fn, metrics, run_model

# Fields that define the experiment (hashed). Run-control fields (steps, deadline, paths) may change on resume.
HASHED = ("n", "head", "lr", "batch_size", "seed", "blend", "alpha", "warmup", "weight_decay",
          "arch", "l2", "l3", "k", "router", "balance", "init_from")


def parse_duration(s: str) -> float:
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s)
    if not m or not any(m.groups()):
        raise ValueError(f"bad duration {s!r}")
    h, mi, se = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + se


def config_hash(cfg: dict) -> str:
    # N1 runs hash exactly as before the N2/N3 fields existed (their defaults are left out).
    keys = HASHED if cfg.get("arch", "n1") != "n1" else HASHED[:9]
    return hashlib.sha256(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()[:16]


def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def find_resume(run_dir: str, run_id: str):
    cands = [run_dir] + sorted(glob.glob(f"/kaggle/input/*/ckpt/run-{run_id}"))
    repo = os.environ.get("HF_REPO")
    if repo:
        try:
            from huggingface_hub import snapshot_download
            cands.append(os.path.join(snapshot_download(repo, allow_patterns=[f"ckpt/run-{run_id}/*"]), "ckpt", f"run-{run_id}"))
        except Exception as e:      # noqa: BLE001 - resume from the other locations
            print("HF resume unavailable:", e)
    best = None
    for d in cands:
        for p in glob.glob(os.path.join(d, "step-*.pt")):
            step = int(re.search(r"step-(\d+)\.pt", p).group(1))
            if best is None or step > best[0]:
                best = (step, p)
    return best[1] if best else None


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    t = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def evaluate(model, val_paths, cfg, device, max_batches=20):
    stream = ShardStream(val_paths, cfg["batch_size"], seed=1234, weights=tuple(cfg["blend"]), split=VAL, prefetch=0,
                         arch=cfg.get("arch", "n1"))
    tot_ll = tot_br = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for b in stream:
            out = run_model(model, b, device)
            ll, br = metrics(model, out, torch.from_numpy(b["target"]).to(device), torch.from_numpy(b["escore"]).to(device))
            tot_ll += ll; tot_br += br; n += 1
            if n >= max_batches:
                break
    model.train()
    return (tot_ll / n, tot_br / n) if n else (float("nan"), float("nan"))


def train(cfg: dict, resume: str = "auto") -> dict:
    run_dir = os.path.join(cfg["ckpt_dir"], f"run-{cfg['run_id']}")
    os.makedirs(run_dir, exist_ok=True)
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    # Shared machines (run control, not part of the config hash): cap this process's share of GPU memory and CPU threads.
    if device == "cuda" and os.environ.get("GPU_MEM_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_MEM_FRACTION"]))
    if os.environ.get("TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
    chash = config_hash(cfg)
    train_paths = sorted(glob.glob(cfg["train_glob"]))
    val_paths = sorted(glob.glob(cfg["val_glob"])) if cfg.get("val_glob") else train_paths
    if not train_paths:
        raise FileNotFoundError(cfg["train_glob"])

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    model = build_model(cfg)
    if cfg.get("init_from") and isinstance(model, NNUE2):      # sparse upcycling of a trained N2 into K experts
        from .export import load_checkpoint_model
        model.upcycle_from(load_checkpoint_model(cfg["init_from"]))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    use_amp = device == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, cursor, best_val, wall_used = 0, None, float("inf"), 0.0

    path = None if resume == "none" else (find_resume(run_dir, cfg["run_id"]) if resume == "auto" else resume)
    if path:
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck["config_hash"] != chash and not cfg.get("allow_config_change"):
            raise SystemExit(f"config hash mismatch ({ck['config_hash']} vs {chash}); pass --allow-config-change")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        step, cursor, best_val, wall_used = ck["step"], ck["cursor"], ck["best_val"], ck["wall"]
        random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if device == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng"]["cuda"]])
        print(f"resumed from {path} at step {step}")
    else:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            json.dump({**cfg, "config_hash": chash}, f, indent=1)    # JSON is valid YAML

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    deadline = parse_duration(cfg["deadline"]) if cfg.get("deadline") else None
    t_start = time.time()
    wall = lambda: wall_used + (time.time() - t_start)    # noqa: E731

    def save(cursor_):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
              "cursor": cursor_, "best_val": best_val, "wall": wall(), "config_hash": chash,
              "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None}}
        atomic_save(ck, os.path.join(run_dir, f"step-{step}.pt"))
        for old in sorted(glob.glob(os.path.join(run_dir, "step-*.pt")), key=lambda p: int(re.search(r"(\d+)\.pt", p).group(1)))[:-3]:
            os.remove(old)

    metrics_f = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    stream = ShardStream(train_paths, cfg["batch_size"], seed=cfg["seed"], weights=tuple(cfg["blend"]), cursor=cursor,
                         arch=cfg.get("arch", "n1"), workers=cfg.get("loader_workers") or 0)
    last_ckpt = time.time()
    seen = 0
    model.train()
    cur = cursor
    for b in stream:
        if step >= cfg["steps"] or (cfg.get("stop_after") and step >= cfg["stop_after"]) or stop["flag"] or (deadline and wall() >= deadline):
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        target, esc = torch.from_numpy(b["target"]).to(device), torch.from_numpy(b["escore"]).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = run_model(model, b, device)
            loss = loss_fn(model, out.float(), target, esc, cfg["alpha"])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        model.clamp_()
        step += 1; seen += len(esc); cur = b["cursor"]
        rec = {"step": step, "loss": float(loss.detach()), "lr": opt.param_groups[0]["lr"], "positions": step * cfg["batch_size"]}
        if step % cfg["val_every"] == 0:
            ll, br = evaluate(model, val_paths, cfg, device)
            rec.update(val_logloss=ll, val_brier=br)
            if ll < best_val:
                best_val = ll
                atomic_save({"model": model.state_dict(), "step": step, "val_logloss": ll, "cfg": cfg}, os.path.join(run_dir, "best.pt"))
        metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
        if time.time() - last_ckpt >= cfg["ckpt_every_s"] or (cfg.get("ckpt_every_steps") and step % cfg["ckpt_every_steps"] == 0):
            save(cur); last_ckpt = time.time()
    save(cur)      # final checkpoint (deadline, SIGTERM, or end of run)
    metrics_f.close()
    final = os.path.join(run_dir, "final.pt")
    atomic_save({"model": model.state_dict(), "step": step, "cfg": cfg}, final)
    if os.environ.get("HF_REPO"):
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(folder_path=run_dir, path_in_repo=f"ckpt/run-{cfg['run_id']}", repo_id=os.environ["HF_REPO"])
        except Exception as e:      # noqa: BLE001
            print("HF upload failed:", e)
    return {"step": step, "run_dir": run_dir, "final": final, "stopped_by_signal": stop["flag"]}


DEFAULTS = dict(run_id="dev", ckpt_dir="ckpt", loader_workers=0, arch="n1", l2=16, l3=32, k=1, router="none", balance=0.01, init_from=None,
                n=256, head=1, lr=1e-3, weight_decay=0.0, batch_size=4096, steps=1000,
                warmup=100, seed=1, blend=[0.6, 0.25, 0.15], alpha=0.0, val_every=200, ckpt_every_s=1200,
                ckpt_every_steps=0, deadline=None, train_glob="data/train/*.bin", val_glob=None, device=None)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON/YAML file overriding the defaults")
    ap.add_argument("--resume", default="auto", help="auto | none | path to a step-N.pt")
    ap.add_argument("--allow-config-change", action="store_true")
    ap.add_argument("--deadline")
    for k, v in DEFAULTS.items():
        if k not in ("blend", "deadline"):
            ap.add_argument("--" + k.replace("_", "-"), type=(type(v) if v is not None else str), default=None)
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        import yaml
        cfg.update(yaml.safe_load(open(a.config)))
    for k in DEFAULTS:
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    cfg["deadline"] = a.deadline or cfg.get("deadline")
    cfg["allow_config_change"] = a.allow_config_change
    print(json.dumps(train(cfg, a.resume)))
