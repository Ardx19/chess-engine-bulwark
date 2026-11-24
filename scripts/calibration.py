"""Outcome calibration of the engine's W/D/L in a declared setting (B.1 item 5, B.6 "probability quality").

Input: save-rate / suite JSONL files from scripts/saverate.py run with a WDL net and UCI_ShowWDL (each game has a
per-move trace [ply, W, D, L (per mille, engine's view), cp] and the final result from the engine's view).
Predictions within one game are correlated, so metrics are reported twice: over all moves, and over one prediction
per game (the first move, i.e. the starting position).

Metrics: 3-class log loss, 3-class Brier score, and ECE of the expected score E = W + D/2 against the game score with
equal-width bins (default 10); plus per-class reliability tables, optionally drawn as a reliability diagram.
The setting (engine build, net, budgets, opponent) is part of the result: there is no opponent-independent truth.

Usage: python3 scripts/calibration.py results/saverate/test-*.jsonl [--bins 10] [--png results/reliability.png]
"""
from __future__ import annotations

import argparse
import json
import math

OUTCOME = {"W": (1, 0, 0), "D": (0, 1, 0), "L": (0, 0, 1)}
SCORE = {"W": 1.0, "D": 0.5, "L": 0.0}


def collect(paths):
    all_moves, first = [], []
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                g = json.loads(line)
                preds = [(r[1] / 1000, r[2] / 1000, r[3] / 1000) for r in g.get("trace", []) if r[1] is not None]
                if not preds:
                    continue
                all_moves += [(p_, g["result"]) for p_ in preds]
                first.append((preds[0], g["result"]))
    return all_moves, first


def metrics(rows, bins: int):
    if not rows:
        return None
    eps = 1e-6
    ll = br = 0.0
    edges = [[0, 0.0, 0.0] for _ in range(bins)]           # count, sum predicted E, sum actual score
    cls = [[[0, 0.0, 0.0] for _ in range(bins)] for _ in range(3)]
    for (w, d, l), res in rows:
        t = OUTCOME[res]
        p = (w, d, l)
        ll -= math.log(max(p[t.index(1)], eps))
        br += sum((pi - ti) ** 2 for pi, ti in zip(p, t))
        e = w + d / 2
        k = min(int(e * bins), bins - 1)
        edges[k][0] += 1; edges[k][1] += e; edges[k][2] += SCORE[res]
        for c in range(3):
            kc = min(int(p[c] * bins), bins - 1)
            cls[c][kc][0] += 1; cls[c][kc][1] += p[c]; cls[c][kc][2] += t[c]
    n = len(rows)
    ece = sum(abs(s_p - s_a) for c, s_p, s_a in edges if c) / n
    rel = [{"bin": [i / bins, (i + 1) / bins], "n": c, "mean_pred": s_p / c, "mean_score": s_a / c}
           for i, (c, s_p, s_a) in enumerate(edges) if c]
    per_class = {name: [{"bin": [i / bins, (i + 1) / bins], "n": c, "mean_pred": s_p / c, "freq": s_a / c}
                        for i, (c, s_p, s_a) in enumerate(cls[k]) if c] for k, name in enumerate("WDL")}
    return {"n": n, "log_loss": ll / n, "brier": br / n, "ece_expected_score": ece, "bins": bins,
            "reliability_expected_score": rel, "reliability_per_class": per_class}


def plot(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    m = rep["all_moves"]
    series = [("expected score", m["reliability_expected_score"], "mean_score")] + \
             [(f"P({k})", m["reliability_per_class"][k], "freq") for k in "WDL"]
    for ax, (title, rows, key) in zip(axes, series):
        ax.plot([0, 1], [0, 1], color="#999", lw=1)
        ax.plot([r["mean_pred"] for r in rows], [r[key] for r in rows], marker="o")
        ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("predicted"); ax.set_ylabel("observed")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--png")
    ap.add_argument("--json")
    a = ap.parse_args()
    all_moves, first = collect(a.files)
    rep = {"files": a.files, "games": len(first), "all_moves": metrics(all_moves, a.bins),
           "first_move_per_game": metrics(first, a.bins)}
    print(json.dumps({k: (v if k not in ("all_moves", "first_move_per_game") or v is None else
                          {x: v[x] for x in ("n", "log_loss", "brier", "ece_expected_score")}) for k, v in rep.items()}, indent=1))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    if a.png and rep["all_moves"]:
        plot(rep, a.png)


if __name__ == "__main__":
    main()
