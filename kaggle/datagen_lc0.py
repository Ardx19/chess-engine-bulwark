"""Large-scale dataset builder with Lc0 labels only (no Stockfish), for one Kaggle CPU session. No GPU, no training.

Lc0's training files already carry Lc0's searched W/D/L (root q, d) and the game result, so this only downloads,
converts and splits: download (several tars in parallel, the bottleneck) -> lc0_convert (filters + 3-5 piece
tablebase override) -> 40-byte training records split by game id (train / val / test) -> shards of ~2M records.
Label blend at training time: Lc0 0.6 + result 0.15, renormalised = 0.8 / 0.2 (nnue/targets.py).

Output, sized for Kaggle's 20 GB limit: <out>/bulwark_data/s<S>/training/{train,val,test}/s<S>-<NNNNN>.bin plus
state.json and report.json. A shard and its tars are recorded in state.json together, so a session that ends early
loses only the tars still in memory; they are redone next round. A new round needs only the previous state.json
(found under --restore-root at any depth); it writes new shard numbers, so the outputs of all rounds are merged by
collecting their training/ folders (scripts/merge_kaggle_data.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import shutil
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline.convert import convert                                      # noqa: E402
from pipeline.records import TEST, TRAIN, VAL, read_shard, split_of, to_training, write_shard   # noqa: E402
from pipeline.run import BASE, download, list_tars, pick_tars             # noqa: E402

SPLITS = {TRAIN: "train", VAL: "val", TEST: "test"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_state(d, state):
    tmp = os.path.join(d, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, os.path.join(d, "state.json"))


def restore_state(d, session, roots):
    best = None
    for root in roots:
        for sf in glob.glob(os.path.join(root, "**", "bulwark_data", f"s{session}", "state.json"), recursive=True):
            if os.path.dirname(os.path.abspath(sf)) == os.path.abspath(d):
                continue
            st = json.load(open(sf))
            if st.get("mode") != "lc0":
                continue
            if best is None or st["positions"] > best[0]["positions"]:
                best = (st, sf)
    if best:
        log(f"continuing from {best[1]}: {best[0]['positions']:,} positions, {len(best[0]['done_tars'])} tars done")
        return best[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--num-sessions", type=int, default=5)
    ap.add_argument("--target", type=int, default=300_000_000, help="positions for this session (all rounds)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--restore-root", action="append", default=[])
    ap.add_argument("--converter", required=True)
    ap.add_argument("--syzygy")
    ap.add_argument("--shard-records", type=int, default=2_000_000, help="train shard size")
    ap.add_argument("--heldout-records", type=int, default=500_000, help="val/test shard size")
    ap.add_argument("--downloads", type=int, default=3, help="parallel tar downloads")
    ap.add_argument("--start", type=float, default=time.time())
    ap.add_argument("--deadline-h", type=float, default=11.0)
    ap.add_argument("--max-output-gb", type=float, default=18.0, help="stop before this much output (Kaggle: 20 GB)")
    ap.add_argument("--run", default="test91")
    ap.add_argument("--tars", nargs="*", help="explicit tar names (tests)")
    ap.add_argument("--max-games", type=int, default=0, help="games per tar (tests)")
    a = ap.parse_args()

    S = a.session
    d = os.path.join(a.out, "bulwark_data", f"s{S}")
    for name in SPLITS.values():
        os.makedirs(os.path.join(d, "training", name), exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    state = restore_state(d, S, a.restore_root) or {
        "mode": "lc0", "session": S, "num_sessions": a.num_sessions, "run": a.run, "rounds": 0,
        "tars": a.tars or pick_tars(list_tars(a.run), S, a.num_sessions, 10 ** 6, 3),
        "done_tars": [], "next_shard": 0, "positions": 0, "counts": {n: 0 for n in SPLITS.values()}, "convert_totals": {}}
    state["rounds"] += 1
    save_state(d, state)
    stop_at = a.start + a.deadline_h * 3600
    out_bytes = 0
    log(f"session {S} round {state['rounds']}: {state['positions']:,} positions so far, target {a.target:,}, "
        f"{(stop_at - time.time()) / 3600:.1f} h, {a.downloads} parallel downloads")

    done = set(state["done_tars"])
    todo = [t for t in state["tars"] if t not in done]
    q: queue.Queue = queue.Queue(maxsize=a.downloads + 1)
    stop = threading.Event()

    def downloader(names):
        try:
            for name in names:
                if stop.is_set():
                    break
                path = os.path.join(a.tmp, name)
                try:
                    dt = download(f"{BASE}{a.run}/{name}", path) if a.max_games == 0 or not os.path.exists(path) else 0.0
                    q.put((name, path, dt))
                except Exception as e:      # noqa: BLE001 - skip a tar that will not download
                    log(f"download failed, skipping {name}: {e}")
        finally:
            q.put(None)                     # always signal the end, also when stopping early (target/deadline)

    workers = [threading.Thread(target=downloader, args=(todo[i::a.downloads],), daemon=True) for i in range(a.downloads)]
    for w in workers:
        w.start()

    buf = {n: [] for n in SPLITS.values()}
    buf_n = {n: 0 for n in SPLITS.values()}
    pending = []                        # tars whose records are still (partly) in the buffers

    def flush(force=False):
        nonlocal out_bytes
        wrote = False
        for name, limit in (("train", a.shard_records), ("val", a.heldout_records), ("test", a.heldout_records)):
            while buf_n[name] >= limit or (force and buf_n[name] > 0):
                arr = np.concatenate(buf[name])
                take, rest = arr[:limit], arr[limit:]
                path = os.path.join(d, "training", name, f"s{S}-{state['next_shard']:05d}.bin")
                write_shard(path + ".tmp", take, state["next_shard"])
                os.replace(path + ".tmp", path)
                out_bytes += os.path.getsize(path)
                state["next_shard"] += 1
                state["counts"][name] += len(take)
                buf[name] = [rest] if len(rest) else []
                buf_n[name] = len(rest)
                wrote = True
        if wrote and not any(buf_n.values()):       # everything those tars produced is on disk
            state["done_tars"] += pending
            pending.clear()
        if wrote or force:
            save_state(d, state)

    finished = 0
    while finished < len(workers):
        item = q.get()
        if item is None:
            finished += 1
            continue
        name, path, dl = item
        if (time.time() >= stop_at or state["positions"] >= a.target
                or out_bytes / 1e9 >= a.max_output_gb):
            stop.set()
            os.remove(path) if os.path.exists(path) else None
            continue
        idx = state["tars"].index(name)            # stable per tar, so a redone tar gets the same game ids
        full = os.path.join(a.tmp, f"full-{idx}.bin")
        st = convert(path, full, a.converter, session=(S << 12) | idx, shard_id=S * 10000 + idx,
                     syzygy=a.syzygy, max_games=a.max_games)
        rec = read_shard(full)
        os.remove(full)
        if not a.max_games:
            os.remove(path)
        sp = split_of(rec["gid"])
        for code, sname in SPLITS.items():
            part = to_training(rec[sp == code])
            if len(part):
                buf[sname].append(part)
                buf_n[sname] += len(part)
        state["positions"] += len(rec)
        pending.append(name)
        for k, v in st.items():
            if isinstance(v, int):
                state["convert_totals"][k] = state["convert_totals"].get(k, 0) + v
        log(f"{name}: {len(rec):,} positions (download {dl:.0f}s, convert {st['seconds']:.0f}s); "
            f"total {state['positions']:,} / {a.target:,}")
        flush()
    flush(force=True)
    state["done_tars"] += pending
    pending.clear()
    save_state(d, state)
    for f in glob.glob(os.path.join(a.tmp, "*.tar")):
        if not a.max_games:
            os.remove(f)
    report = {"session": S, "round": state["rounds"], "positions": state["positions"], "target": a.target,
              "done": state["positions"] >= a.target, "tars_done": len(state["done_tars"]), "counts": state["counts"],
              "output_gb_this_round": round(out_bytes / 1e9, 2), "convert_totals": state["convert_totals"],
              "labels": "Lc0 root W/D/L + game result (no Stockfish)", "lc0_run": a.run,
              "license": "Lc0 training data is ODbL 1.0 (credit: Leela Chess Zero); derived data stays ODbL."}
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    log(json.dumps({k: report[k] for k in ("session", "round", "positions", "done", "counts", "output_gb_this_round")}))
    if not report["done"]:
        log("NOT DONE: run another round with this output added as input (it continues with the remaining tars)")


if __name__ == "__main__":
    main()
