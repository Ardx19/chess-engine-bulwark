"""Merge pentanomial match results (Kaggle match_runner results.json files and/or fastchess logs) and report Elo with
a 95% interval, normalised Elo (nElo) and the SPRT log-likelihood ratio in fastchess's default normalised model
(bounds in nElo; GSPRT approximation, matches fastchess to ~0.02 when the score is near 50%).

Usage: python3 scripts/merge_penta.py [--elo0 0 --elo1 5] results/kaggle_match/*/results.json results/sprt-x/log.txt
"""
from __future__ import annotations

import argparse
import json
import math
import re


def read(path):
    if path.endswith(".json"):
        return json.load(open(path))["penta"]
    m = re.findall(r"Ptnml\(0-2\): \[(\d+), (\d+), (\d+), (\d+), (\d+)\]", open(path).read())
    return [int(x) for x in m[-1]] if m else [0] * 5


def stats(p):
    n = sum(p)
    s = [0, 0.25, 0.5, 0.75, 1]
    mu = sum(pi * si for pi, si in zip(p, s)) / n
    var = sum(pi * (si - mu) ** 2 for pi, si in zip(p, s)) / n
    return n, mu, var


def elo(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


NELO = 800 / math.log(10) / math.sqrt(2)      # nElo = NELO * (mu - 0.5) / sigma_pair (fastchess, fishtest)


def llr(p, nelo0, nelo1):
    """Normalised GSPRT approximation: LLR = n (t1 - t0) (2 t - t0 - t1) / 2 with t = (mu - 0.5) / sigma_pair."""
    n, mu, var = stats(p)
    if var == 0:
        return 0.0
    t, t0, t1 = (mu - 0.5) / math.sqrt(var), nelo0 / NELO, nelo1 / NELO
    return n * (t1 - t0) * (2 * t - t0 - t1) / 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--elo0", type=float, default=0)
    ap.add_argument("--elo1", type=float, default=5)
    a = ap.parse_args()
    total = [0] * 5
    for f in a.files:
        p = read(f)
        total = [x + y for x, y in zip(total, p)]
        if sum(p):
            n, mu, _ = stats(p)
            print(f"{f}: pairs {n}, penta {p}, Elo {elo(mu):+.1f}")
    n, mu, var = stats(total)
    se = math.sqrt(var / n)
    lo, hi = elo(mu - 1.96 * se), elo(mu + 1.96 * se)
    bound = math.log((1 - 0.05) / 0.05)
    L = llr(total, a.elo0, a.elo1)
    verdict = "H1 (accept)" if L >= bound else "H0 (reject)" if L <= -bound else "continue"
    print(f"TOTAL: {2 * n} games ({n} pairs), penta {total}")
    print(f"Elo {elo(mu):+.1f} (95% {lo:+.1f} .. {hi:+.1f}), nElo {NELO * (mu - 0.5) / math.sqrt(var):+.1f}, LLR {L:.2f} ({-bound:.2f}, {bound:.2f}) "
          f"[{a.elo0}, {a.elo1}]: {verdict}")
