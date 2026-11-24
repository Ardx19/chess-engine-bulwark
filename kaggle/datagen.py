"""Resumable CPU-only dataset builder for one Kaggle session (or any Linux box). No GPU, no training.

Per round (one notebook run):
  restore the previous round's progress -> label any leftover positions -> take more Lc0 tars (download, convert,
  split into pieces of PIECE positions, delete the tar) and label each piece with Stockfish -> stop starting new work
  at the deadline -> rebuild this session's train/ val/ test/ shards from every labeled piece -> report.

Progress is saved after every piece (state.json + one file per piece, written atomically), so a session that ends
early loses at most the piece in flight. The next round continues from the previous round's output: in Kaggle, add
that notebook output as an input; it is found under <restore-root>/*/bulwark_data/s<SESSION>/.

Sessions are disjoint: session S takes every NUM_SESSIONS-th tar of the run (in a fixed random order spread over
the whole run), and its game ids carry the session number, so the outputs of different sessions can be merged by
copying their training/ folders together.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline.convert import convert                                      # noqa: E402
from pipeline.make_training import make_training                          # noqa: E402
from pipeline.records import read_shard, write_shard                      # noqa: E402
from pipeline.relabel import relabel, sha256_file                         # noqa: E402
from pipeline.run import BASE, download, list_tars, pick_tars             # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_state(d, state):
    tmp = os.path.join(d, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, os.path.join(d, "state.json"))


def restore(d, session, roots):
    """Copy the most advanced earlier copy of this session's folder (by round, then labeled positions) into d."""
    best = None
    for root in roots:
        for sf in glob.glob(os.path.join(root, "*", "bulwark_data", f"s{session}", "state.json")) + \
                  glob.glob(os.path.join(root, "bulwark_data", f"s{session}", "state.json")):
            st = json.load(open(sf))
            key = (st.get("rounds", 0), st.get("labeled", 0))
            if os.path.dirname(os.path.abspath(sf)) != os.path.abspath(d) and (best is None or key > best[0]):
                best = (key, os.path.dirname(sf))
    if best:
        mine = os.path.join(d, "state.json")
        if os.path.exists(mine) and (json.load(open(mine)).get("rounds", 0), json.load(open(mine)).get("labeled", 0)) >= best[0]:
            return None
        log(f"restoring progress from {best[1]}")
        for sub in ("master", "labeled"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
            for f in glob.glob(os.path.join(best[1], sub, "*.bin")):
                dst = os.path.join(d, sub, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy2(f, dst)
        shutil.copy2(os.path.join(best[1], "state.json"), os.path.join(d, "state.json"))
        return best[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--num-sessions", type=int, default=5)
    ap.add_argument("--target", type=int, default=10_000_000, help="positions to label in this session (all rounds)")
    ap.add_argument("--out", required=True, help="persistent output root (bulwark_data/ is created inside)")
    ap.add_argument("--tmp", required=True, help="scratch dir for tars (not kept)")
    ap.add_argument("--restore-root", action="append", default=[], help="where earlier outputs are mounted")
    ap.add_argument("--converter", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--syzygy", help="3-5 piece tables for the tablebase label override")
    ap.add_argument("--sf-nodes", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--piece", type=int, default=50_000, help="positions per labeled piece (the checkpoint unit)")
    ap.add_argument("--start", type=float, default=time.time(), help="session start (epoch seconds)")
    ap.add_argument("--deadline-h", type=float, default=11.0, help="no new piece/tar after this many hours from --start")
    ap.add_argument("--final-min", type=float, default=20.0, help="minutes reserved for the final training-shard build")
    ap.add_argument("--run", default="test91")
    ap.add_argument("--tars", nargs="*", help="explicit tar names (tests)")
    ap.add_argument("--max-games", type=int, default=0, help="games per tar (tests)")
    a = ap.parse_args()

    S = a.session
    d = os.path.join(a.out, "bulwark_data", f"s{S}")
    os.makedirs(os.path.join(d, "master"), exist_ok=True)
    os.makedirs(os.path.join(d, "labeled"), exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    restore(d, S, a.restore_root)
    sp = os.path.join(d, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {
        "session": S, "num_sessions": a.num_sessions, "run": a.run, "sf_nodes": a.sf_nodes, "rounds": 0,
        "tars": a.tars or pick_tars(list_tars(a.run), S, a.num_sessions, 10 ** 6, 3),
        "done_tars": {}, "pieces": {}, "kept": 0, "labeled": 0, "convert_totals": {}}
    if state["session"] != S or state["sf_nodes"] != a.sf_nodes:
        sys.exit("state belongs to a different session or node budget")
    state["rounds"] += 1
    save_state(d, state)
    stop_at = a.start + a.deadline_h * 3600 - a.final_min * 60
    time_left = lambda: stop_at - time.time()           # noqa: E731
    sf_hash = sha256_file(a.stockfish)
    log(f"session {S} round {state['rounds']}: {state['labeled']:,} labeled, {state['kept']:,} converted, "
        f"target {a.target:,}, {time_left() / 3600:.1f} h of work time, {a.workers} workers")

    def label_pending():
        for name in sorted(n for n, s in state["pieces"].items() if s == "master"):
            if time_left() <= 0 or state["labeled"] >= a.target:
                return False
            src, dst = os.path.join(d, "master", name), os.path.join(d, "labeled", name)
            t0 = time.time()
            info = relabel(src, dst + ".tmp", a.stockfish, a.sf_nodes, a.workers)
            os.replace(dst + ".tmp", dst)
            os.remove(src)
            state["pieces"][name] = "labeled"
            state["labeled"] += info["records"]
            state["sf"] = {"name": info["stockfish"], "sha256": sf_hash, "nodes": a.sf_nodes}
            save_state(d, state)
            rate = info["records"] / max(time.time() - t0, 1e-9)
            log(f"labeled {name}: {info['records']:,} positions at {rate:.0f}/s; total {state['labeled']:,}/{a.target:,}")
        return True

    ok = label_pending()
    while ok and state["labeled"] < a.target and time_left() > 0:
        todo = [t for t in state["tars"] if t not in state["done_tars"]]
        if not todo:
            log("no tars left in this session's slice")
            break
        if state["kept"] >= a.target:     # enough converted; only labeling remains
            ok = label_pending()
            break
        name = todo[0]
        idx = len(state["done_tars"])
        tar = os.path.join(a.tmp, name)
        log(f"tar {idx}: {name}")
        dl = download(f"{BASE}{a.run}/{name}", tar) if not os.path.exists(tar) else 0.0
        full = os.path.join(a.tmp, "full.bin")
        st = convert(tar, full, a.converter, session=(S << 12) | idx, shard_id=S * 10000 + idx,
                     syzygy=a.syzygy, max_games=a.max_games)
        rec = read_shard(full)
        gid = rec["gid"]
        cuts, k = [0], a.piece               # pieces of ~PIECE positions, cut only between games
        while k < len(rec):
            while k < len(rec) and gid[k] == gid[k - 1]:
                k += 1
            if k < len(rec):
                cuts.append(k)
            k += a.piece
        cuts.append(len(rec))
        for j in range(len(cuts) - 1):
            pn = f"s{S}-t{idx:04d}-p{j:03d}.bin"
            write_shard(os.path.join(d, "master", pn + ".tmp"), rec[cuts[j]:cuts[j + 1]], S * 10000 + idx)
            os.replace(os.path.join(d, "master", pn + ".tmp"), os.path.join(d, "master", pn))
            state["pieces"][pn] = "master"
        os.remove(full)
        os.remove(tar)
        state["done_tars"][name] = {"kept": int(st.get("kept", len(rec))), "seen": int(st.get("seen", 0)),
                                    "download_s": round(dl, 1), "convert_s": round(st["seconds"], 1)}
        state["kept"] += len(rec)
        for k, v in st.items():
            if isinstance(v, int):
                state["convert_totals"][k] = state["convert_totals"].get(k, 0) + v
        save_state(d, state)
        log(f"converted {len(rec):,} positions ({st.get('seen', 0):,} seen) in {st['seconds']:.0f}s, download {dl:.0f}s")
        ok = label_pending()

    # Training shards from everything labeled so far (rebuilt each round; prefix keeps sessions apart).
    labeled = sorted(glob.glob(os.path.join(d, "labeled", "*.bin")))
    tdir = os.path.join(d, "training")
    shutil.rmtree(tdir, ignore_errors=True)
    counts = make_training(labeled, tdir, shard_records=1_000_000, prefix=f"s{S}") if labeled else {}
    report = {"session": S, "round": state["rounds"], "labeled": state["labeled"], "converted": state["kept"],
              "target": a.target, "done": state["labeled"] >= a.target, "tars_done": len(state["done_tars"]),
              "pending_unlabeled_pieces": sum(1 for s in state["pieces"].values() if s == "master"),
              "training_counts": counts, "stockfish": state.get("sf"), "convert_totals": state["convert_totals"],
              "lc0_run": a.run, "license": "Lc0 training data is ODbL 1.0 (credit: Leela Chess Zero); derived data stays ODbL."}
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    log(json.dumps({k: report[k] for k in ("session", "round", "labeled", "converted", "done", "training_counts")}))
    if not report["done"]:
        log("NOT DONE: run another round with this output added as input (it resumes where this one stopped)")


if __name__ == "__main__":
    main()
