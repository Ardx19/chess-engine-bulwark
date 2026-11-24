"""Fortress recognition (B.6 fortress suite, metric a): the share of suite positions the engine evaluates as a draw
within a fixed budget: P(draw) > 0.6 (WDL net, UCI_ShowWDL) or |score| < 50 internal units.
Metric b (held in play) uses scripts/saverate.py on the same EPD.

Suite format: EPD with `c0 "worse=<w|b> ..."` naming the defending side (the curated suite lives in
benchsets/fortress/{dev,test}.epd; sources: CPW Fortress page, Guid & Bratko 2012, drawn imbalanced game positions).

Usage: python3 scripts/fortress_eval.py benchsets/fortress/dev.epd --engine build/bulwark --eopt EvalFile=... --nodes 200000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saverate import parse_set      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--engine", default="build/bulwark")
    ap.add_argument("--eopt", action="append", default=[])
    ap.add_argument("--nodes", type=int, default=200000)
    ap.add_argument("--draw-p", type=float, default=0.6)
    ap.add_argument("--draw-cp", type=int, default=50)
    ap.add_argument("--json")
    a = ap.parse_args()
    eng = chess.engine.SimpleEngine.popen_uci(a.engine)
    opts = {"UCI_ShowWDL": True}
    for kv in a.eopt:
        k, v = kv.split("=", 1)
        opts[k] = v
    eng.configure({k: v for k, v in opts.items() if k in eng.options})
    rows = []
    for t in parse_set(a.suite):
        b = chess.Board(t["fen"])
        info = eng.analyse(b, chess.engine.Limit(nodes=a.nodes), game=object())
        cp = info["score"].pov(b.turn).score(mate_score=32000)
        wdl = info.get("wdl")
        pd = wdl.pov(b.turn).draws / 1000 if wdl is not None else None
        rec = (pd is not None and pd > a.draw_p) or abs(cp) < a.draw_cp
        rows.append({"id": t["id"], "cp": cp, "p_draw": pd, "recognized": rec})
    eng.quit()
    n = len(rows)
    rep = {"suite": a.suite, "positions": n, "recognized": sum(r["recognized"] for r in rows),
           "rate": sum(r["recognized"] for r in rows) / n if n else None, "nodes": a.nodes, "rows": rows}
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
