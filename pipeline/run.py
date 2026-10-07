"""Pilot / production data round (one Kaggle CPU session = one shard id).

  download tar -> convert + filter (+ tablebase override) -> delete tar -> ... -> Stockfish relabel -> manifest

Deadline-aware: no new tars after `--stop-taking-h` (default 10.5 h of a 12 h session); the manifest is written
last. Every stage is timed so the pilot report can size the production round (B.9).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

from .convert import convert
from .manifest import write_manifest
from .records import SF_WD_ABSENT, read_shard, write_shard
from .relabel import relabel

BASE = "https://storage.lczero.org/files/training_data/"


def list_tars(run: str) -> list[str]:
    html = urllib.request.urlopen(f"{BASE}{run}/", timeout=60).read().decode()
    return sorted(set(re.findall(r'href="(training-[^"]+\.tar)"', html)))


def download(url: str, dest: str, retries: int = 4, throttle_s: float = 0.0) -> float:
    t0 = time.time()
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    blk = r.read(1 << 20)
                    if not blk:
                        break
                    f.write(blk)
            time.sleep(throttle_s)
            return time.time() - t0
        except Exception as e:                      # noqa: BLE001
            print(f"download retry {attempt}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="test91", help="Lc0 training-data run directory")
    ap.add_argument("--session", type=int, default=0, help="session id (also selects this session's tar slice)")
    ap.add_argument("--num-sessions", type=int, default=1)
    ap.add_argument("--max-tars", type=int, default=2, help="pilot: 2-5 tars")
    ap.add_argument("--skip-tars", type=int, default=0, help="skip the newest N tars (may be still uploading)")
    ap.add_argument("--local-tars", nargs="*", help="use these local tar files instead of downloading")
    ap.add_argument("--out", default="data")
    ap.add_argument("--work", default=None)
    ap.add_argument("--converter", default="build/lc0_convert")
    ap.add_argument("--stockfish")
    ap.add_argument("--sf-nodes", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--syzygy")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--stop-taking-h", type=float, default=10.5)
    ap.add_argument("--relabel-limit", type=int, default=0, help="pilot: relabel only the first N positions per shard")
    ap.add_argument("--throttle-s", type=float, default=1.0)
    a = ap.parse_args()

    t_start = time.time()
    work = a.work or os.path.join(a.out, "work")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.join(a.out, "master"), exist_ok=True)
    report = {"run": a.run, "session": a.session, "tars": [], "t_download_s": 0.0, "t_convert_s": 0.0, "bytes_downloaded": 0}

    if a.local_tars:
        sources = [(os.path.basename(t), t) for t in a.local_tars]
    else:
        names = list_tars(a.run)
        names = names[:-a.skip_tars] if a.skip_tars else names
        mine = names[a.session::a.num_sessions][:a.max_tars]
        sources = [(n, None) for n in mine]

    master_shards, files, totals = [], [], {}
    for i, (name, local) in enumerate(sources):
        if (time.time() - t_start) / 3600 > a.stop_taking_h:
            print("deadline: not taking more tars")
            break
        path = local
        if path is None:
            path = os.path.join(work, name)
            dt = download(f"{BASE}{a.run}/{name}", path, throttle_s=a.throttle_s)
            sz = os.path.getsize(path)
            report["t_download_s"] += dt
            report["bytes_downloaded"] += sz
        out = os.path.join(a.out, "master", f"s{a.session:03d}-{i:04d}.bin")
        st = convert(path, out, a.converter, session=(a.session << 12) | i, shard_id=a.session * 1000 + i, stride=a.stride, syzygy=a.syzygy)
        report["t_convert_s"] += st["seconds"]
        report["tars"].append({"name": name, **st})
        for k in ("games", "seen", "kept", "early", "extreme_q", "best_cap_promo", "in_check", "malformed", "tb_override", "stride_skipped", "bad_version", "deleted", "nan"):
            totals[k] = totals.get(k, 0) + st.get(k, 0)
        master_shards.append(out)
        files.append(name)
        if local is None:
            os.remove(path)

    sf_info = None
    if a.stockfish:
        final = []
        for m in master_shards:
            o = m.replace("/master/", "/labeled/")
            os.makedirs(os.path.dirname(o), exist_ok=True)
            info = relabel(m, o, a.stockfish, a.sf_nodes, a.workers, syzygy=None, limit=a.relabel_limit)
            sf_info = info
            final.append(o)
            report.setdefault("relabel", []).append(info)
        shards = final
        # label sanity on what was just produced: Lc0 vs Stockfish expected score
        r = np.concatenate([read_shard(s) for s in shards])
        ok = r["sf_w"] != SF_WD_ABSENT
        if ok.any():
            lc = (1 + r["lc_q"][ok] / 1e4) / 2
            sf = r["sf_w"][ok] / 1000 + r["sf_d"][ok] / 2000
            report["label_sanity"] = {"n": int(ok.sum()), "corr_expected_score": float(np.corrcoef(lc, sf)[0, 1]),
                                      "mean_abs_diff": float(np.abs(lc - sf).mean()),
                                      "draw_mean_lc0": float((r["lc_d"][ok] / 1e4).mean()), "draw_mean_sf": float((r["sf_d"][ok] / 1000).mean())}
    else:
        shards = master_shards

    report["totals"] = totals
    report["acceptance"] = totals.get("kept", 0) / max(1, totals.get("seen", 1))
    report["convert_positions_per_s"] = totals.get("seen", 0) / max(1e-9, report["t_convert_s"])
    report["bytes_per_stored_position"] = 56
    report["wall_s"] = time.time() - t_start
    manifest = write_manifest(os.path.join(a.out, f"manifest-s{a.session:03d}.json"), lc0_run=a.run, files=files,
                              filter_settings={"stride": a.stride, "min_history_plies": 8, "max_abs_q": 0.98,
                                               "skip": ["in_check", "best_move_capture_or_promotion"], "tb_override": bool(a.syzygy)},
                              counts=totals, stockfish=({k: sf_info[k] for k in ("stockfish", "stockfish_sha256", "nodes", "hash_mb", "threads", "tb")} if sf_info else None),
                              shards=shards, extra={"session": a.session, "report": report})
    with open(os.path.join(a.out, f"pilot_report-s{a.session:03d}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("totals", "acceptance", "convert_positions_per_s", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
