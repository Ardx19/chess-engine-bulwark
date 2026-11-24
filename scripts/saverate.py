"""Save-rate harness (B.6): the engine always plays the worse side of each position against Stockfish.

No resign or draw adjudication. Games end by the rules (mate, stalemate, insufficient material, threefold, 50 moves)
or by 3-5 piece Syzygy adjudication: a tablebase draw, cursed win or blessed loss is a draw; a win counts only if
|DTZ| + the halfmove clock fits inside the 50-move limit, otherwise it is a draw. A ply cap ends the game as a draw
with reason "cap" (reported separately; it should be rare).

Each position is played once (the worse side is fixed), so two builds are compared paired by position
(scripts/saverate_stats.py). Output is JSONL, one game per line, resumable (finished ids are skipped). The engine's
per-move W/D/L (when it reports `wdl`, e.g. UCI_ShowWDL with a WDL net) is kept for the calibration study.

Also used for the fortress and tablebase suites (same EPD format; `worse=` names the defending side).

Usage:
  python3 scripts/saverate.py --set benchsets/dev.epd --engine build/bulwark --eopt EvalFile=nets/n1wdl.nnue \
      --eopt DefenseMode=true --enodes 50000 --sf $SF --sfnodes 25000 --out results/saverate/dev-d2.jsonl
  (timed: --etc 8+0.08 instead of --enodes; Stockfish keeps a node limit)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import chess
import chess.engine
import chess.syzygy

_G: dict = {}


def parse_set(path: str):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            b, ops = chess.Board.from_epd(line)
            meta = dict(kv.split("=", 1) for kv in str(ops.get("c0", "")).split() if "=" in kv)
            out.append({"id": ops.get("id") or b.epd(), "fen": b.fen(), "worse": meta.get("worse", "w" if b.turn else "b"),
                        "cluster": meta.get("cluster", ops.get("id") or b.epd())})
    return out


def parse_tc(tc: str):
    base, _, inc = tc.partition("+")
    return float(base), float(inc or 0)


def _init(cfg):
    _G["cfg"] = cfg
    eng = chess.engine.SimpleEngine.popen_uci(cfg["engine"])
    opts = {"Hash": 16, "UCI_ShowWDL": True}
    if cfg.get("syzygy"):
        opts["SyzygyPath"] = cfg["syzygy"]
    for kv in cfg["eopt"]:
        k, v = kv.split("=", 1)
        opts[k] = v
    eng.configure({k: v for k, v in opts.items() if k in eng.options})
    sf = chess.engine.SimpleEngine.popen_uci(cfg["sf"])
    sopts = {"Threads": 1, "Hash": 16}
    if cfg.get("syzygy"):
        sopts["SyzygyPath"] = cfg["syzygy"]
    for kv in cfg["sfopt"]:
        k, v = kv.split("=", 1)
        sopts[k] = v
    sf.configure(sopts)
    _G["eng"], _G["sf"] = eng, sf
    _G["tb"] = chess.syzygy.open_tablebase(cfg["syzygy"]) if cfg.get("syzygy") else None


def tb_adjudicate(b: chess.Board):
    """Result string ('1-0', '0-1', '1/2-1/2') from 3-5 piece tables, or None."""
    tb = _G.get("tb")
    if tb is None or chess.popcount(b.occupied) > 5 or b.castling_rights:
        return None
    try:
        wdl = tb.probe_wdl(b)
        if abs(wdl) < 2:
            return "1/2-1/2"
        dtz = tb.probe_dtz(b)
    except (KeyError, chess.syzygy.MissingTableError):
        return None
    if abs(dtz) + b.halfmove_clock > 100:
        return "1/2-1/2"
    stm_wins = wdl > 0
    return "1-0" if stm_wins == (b.turn == chess.WHITE) else "0-1"


def play(task):
    cfg, eng, sf = _G["cfg"], _G["eng"], _G["sf"]
    b = chess.Board(task["fen"])
    me = chess.WHITE if task["worse"] == "w" else chess.BLACK
    clock = {}
    if cfg.get("etc"):
        base, inc = parse_tc(cfg["etc"])
        clock = {"t": base, "inc": inc}
    trace, reason, result, time_loss = [], None, None, False
    game_key = object()
    start_ply = b.ply()
    while True:
        if b.is_checkmate():
            result, reason = ("0-1" if b.turn == chess.WHITE else "1-0"), "mate"
        elif b.is_stalemate():
            result, reason = "1/2-1/2", "stalemate"
        elif b.is_insufficient_material():
            result, reason = "1/2-1/2", "material"
        elif b.is_repetition(3):
            result, reason = "1/2-1/2", "repetition"
        elif b.halfmove_clock >= 100:
            result, reason = "1/2-1/2", "fifty"
        else:
            r = tb_adjudicate(b)
            if r:
                result, reason = r, "tablebase"
            elif b.ply() - start_ply >= cfg["max_plies"]:
                result, reason = "1/2-1/2", "cap"
        if result:
            break
        if b.turn == me:
            if clock:
                lim = chess.engine.Limit(white_clock=clock["t"] if me == chess.WHITE else 60,
                                         black_clock=clock["t"] if me == chess.BLACK else 60,
                                         white_inc=clock["inc"], black_inc=clock["inc"])
            else:
                lim = chess.engine.Limit(nodes=cfg["enodes"])
            t0 = time.monotonic()
            res = eng.play(b, lim, game=game_key, info=chess.engine.INFO_SCORE)
            if clock:
                clock["t"] -= time.monotonic() - t0
                if clock["t"] < 0:
                    time_loss = True
                    result, reason = ("0-1" if me == chess.WHITE else "1-0"), "time"
                    break
                clock["t"] += clock["inc"]
            info = res.info or {}
            row = [b.ply()]
            wdl = info.get("wdl")
            row += list(wdl.pov(me)) if wdl is not None else [None, None, None]
            sc = info.get("score")
            row.append(sc.pov(me).score(mate_score=32000) if sc is not None else None)
            trace.append(row)
        else:
            res = sf.play(b, chess.engine.Limit(nodes=cfg["sfnodes"]), game=game_key)
        if res.move is None:
            result, reason = "1/2-1/2", "no-move"
            break
        b.push(res.move)
    my = {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(result)
    outcome = "D" if my is None else ("W" if my == me else "L")
    return {**task, "result": outcome, "saved": outcome != "L", "reason": reason, "plies": len(b.move_stack),
            "time_loss": time_loss, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--engine", default="build/bulwark")
    ap.add_argument("--eopt", action="append", default=[], help="engine UCI option NAME=VALUE (repeatable)")
    ap.add_argument("--enodes", type=int, default=50000)
    ap.add_argument("--etc", help="engine time control base+inc in seconds (overrides --enodes)")
    ap.add_argument("--sf", required=True)
    ap.add_argument("--sfopt", action="append", default=[])
    ap.add_argument("--sfnodes", type=int, required=True)
    ap.add_argument("--syzygy", default=os.path.expanduser("~/data/syzygy/tb345"))
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks = parse_set(a.set)
    if a.limit:
        tasks = tasks[:a.limit]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [t for t in tasks if t["id"] not in done]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cfg = {"engine": a.engine, "eopt": a.eopt, "enodes": a.enodes, "etc": a.etc, "sf": a.sf, "sfopt": a.sfopt,
           "sfnodes": a.sfnodes, "syzygy": a.syzygy if os.path.isdir(a.syzygy) else None, "max_plies": a.max_plies}
    with open(a.out + ".config.json", "w") as f:
        json.dump({**cfg, "set": a.set}, f, indent=1)
    saved = n = 0
    with mp.Pool(a.concurrency, initializer=_init, initargs=(cfg,)) as pool, open(a.out, "a") as f:
        for r in pool.imap_unordered(play, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            saved += r["saved"]
            print(f"{n}/{len(todo)} {r['id']} {r['result']} ({r['reason']}, {r['plies']} plies)  saved so far {saved}/{n}", flush=True)


if __name__ == "__main__":
    main()
