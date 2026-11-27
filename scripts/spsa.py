"""SPSA tuning of the search constants in src/tune.h, with fastchess (fishtest-style SPSA).

  run      one SPSA chain: each iteration perturbs every parameter by +/- c_k (random signs), plays PAIRS game pairs
           theta+ vs theta- (same openings, colours swapped) and moves theta along the sign vector by
           r_k * c_k * (wins - losses). State is saved after every iteration and a rerun resumes it.
  average  combine finished chains (e.g. 5 Kaggle sessions + the laptop): mean theta per parameter, optionally
           written back into src/tune.h as the new defaults (--apply).

Schedules as in fishtest: c_k = c_end * (N / k)^0.101, a_k = r_end * c_end^2 * (A + N)^0.602 / (A + k)^0.602,
R_k = a_k / c_k^2 with A = N / 10, where N = planned game pairs and k = pairs played so far.

  python3 scripts/spsa.py run --engine build-tune/bulwark --state results/spsa/laptop.json --hours 3 --seed 7
  python3 scripts/spsa.py average results/spsa/*.json [--apply]

The engine must be a tuning build (cmake -DBULWARK_TUNE=ON); its "tunables" command lists the parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TUNE_H = os.path.join(HERE, "..", "src", "tune.h")


def tunables(engine):
    out = subprocess.run([engine], input="tunables\nquit\n", capture_output=True, text=True).stdout
    ps = {}
    for line in out.splitlines():
        f = line.split()
        if f and f[0] == "tunable":
            ps[f[1]] = {"value": float(f[2]), "lo": int(f[3]), "hi": int(f[4]), "c_end": float(f[5])}
    if not ps:
        raise SystemExit(f"{engine} lists no tunables: build it with -DBULWARK_TUNE=ON")
    return ps


def save(path, st):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(path + ".tmp", path)


def play(a, plus, minus, seed):
    """Play a.pairs game pairs, plus vs minus; return (wins, draws, losses) of plus."""
    def eng(name, vals):
        e = ["-engine", f"cmd={a.engine}", f"name={name}"]
        if a.evalfile:
            e.append(f"option.EvalFile={a.evalfile}")
        return e + [f"option.{k}={v}" for k, v in vals.items()]
    limit = [f"nodes={a.nodes}", "tc=inf"] if a.nodes else [f"tc={a.tc}"]
    cmd = [a.fastchess] + eng("plus", plus) + eng("minus", minus) + [
        "-each", *limit, "option.Hash=16", "option.Threads=1",
        "-openings", f"file={a.book}", "format=pgn", "order=random", "-srand", str(seed),
        "-repeat", "-games", "2", "-rounds", str(a.pairs), "-concurrency", str(a.conc),
        "-resign", "movecount=3", "score=600", "-draw", "movenumber=40", "movecount=8", "score=10",
        "-maxmoves", "250"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.findall(r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)", out)
    if not m:
        raise RuntimeError("no result line from fastchess:\n" + out[-1500:])
    g, w, l, d = (int(x) for x in m[-1])
    return w, d, l


def sround(x, rng):
    """Stochastic rounding: unbiased for small integer parameters."""
    f = math.floor(x)
    return int(f + (rng.random() < x - f))


def run(a):
    params = tunables(a.engine)
    names = [n for n in params if not a.params or n in a.params]
    if os.path.exists(a.state):
        st = json.load(open(a.state))
        print(f"resuming {a.state}: {st['pairs']} pairs", flush=True)
    else:
        st = {"engine_params": params, "names": names, "theta": {n: params[n]["value"] for n in names},
              "start": {n: params[n]["value"] for n in names}, "N": a.total_pairs, "pairs": 0, "iters": 0,
              "wins": 0, "draws": 0, "losses": 0, "seed": a.seed, "nodes": a.nodes, "tc": a.tc, "trace": []}
    rng = random.Random(st["seed"] * 1000003 + st["iters"])
    N, A = st["N"], st["N"] / 10
    deadline = time.time() + a.hours * 3600 if a.hours else None
    it_s = []
    while st["pairs"] < N:
        if deadline and time.time() + (max(it_s) if it_s else 60) * 1.2 > deadline:
            print("time budget used", flush=True)
            break
        k = st["pairs"] + 1
        plus, minus, delta, ck = {}, {}, {}, {}
        for n in st["names"]:
            p = params[n] if n in params else st["engine_params"][n]
            c = p["c_end"] * (N / k) ** 0.101
            ck[n], delta[n] = c, rng.choice((-1, 1))
            th = st["theta"][n]
            plus[n] = min(max(sround(th + c * delta[n], rng), p["lo"]), p["hi"])
            minus[n] = min(max(sround(th - c * delta[n], rng), p["lo"]), p["hi"])
        t0 = time.time()
        w, d, l = play(a, plus, minus, st["seed"] * 100003 + st["iters"])
        it_s.append(time.time() - t0)
        for n in st["names"]:
            p = st["engine_params"][n]
            ak = a.r_end * p["c_end"] ** 2 * (A + N) ** 0.602 / (A + k) ** 0.602
            rk = ak / ck[n] ** 2
            st["theta"][n] = min(max(st["theta"][n] + rk * ck[n] * (w - l) * delta[n], p["lo"]), p["hi"])
        st["pairs"] += a.pairs
        st["iters"] += 1
        st["wins"] += w; st["draws"] += d; st["losses"] += l
        if st["iters"] % 10 == 0:
            st["trace"].append({"pairs": st["pairs"], "theta": dict(st["theta"])})
        save(a.state, st)
        if st["iters"] % a.print_every == 0:
            top = sorted(st["names"], key=lambda n: -abs(st["theta"][n] - st["start"][n]) / st["engine_params"][n]["c_end"])[:5]
            moved = ", ".join(f"{n} {st['start'][n]:g}->{st['theta'][n]:.1f}" for n in top)
            print(f"iter {st['iters']} pairs {st['pairs']}/{N} ({it_s[-1]:.0f}s): {moved}", flush=True)
    print("DONE" if st["pairs"] >= N else "STOPPED", json.dumps({n: round(v, 2) for n, v in st["theta"].items()}),
          flush=True)


def average(a):
    sts = [json.load(open(f)) for f in a.states]
    names = sts[0]["names"]
    pairs = sum(s["pairs"] for s in sts)
    print(f"{len(sts)} chains, {pairs} pairs ({2 * pairs} games)")
    new = {}
    for n in names:
        vals = [s["theta"][n] for s in sts if n in s["theta"]]
        w = [s["pairs"] for s in sts if n in s["theta"]]
        mean = sum(v * x for v, x in zip(vals, w)) / sum(w)
        sd = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        start = sts[0]["start"][n]
        new[n] = int(round(mean))
        flag = "  *" if abs(mean - start) > 2 * sd / max(len(vals), 1) ** 0.5 and new[n] != start else ""
        print(f"{n:14s} {start:8g} -> {mean:9.2f}  (chains sd {sd:6.2f})  new {new[n]}{flag}")
    if a.apply:
        src = open(TUNE_H).read()
        for n, v in new.items():
            src, k = re.subn(rf"X\({n}, -?\d+,", f"X({n}, {v},", src)
            assert k == 1, n
        open(TUNE_H, "w").write(src)
        print(f"wrote the new defaults into {TUNE_H}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", required=True)
    r.add_argument("--state", required=True)
    r.add_argument("--fastchess", default=os.path.expanduser("~/tools/fastchess/fastchess"))
    r.add_argument("--book", default=os.path.expanduser("~/data/books/8moves_v3.pgn"))
    r.add_argument("--evalfile", default="")
    r.add_argument("--nodes", type=int, default=25000, help="fixed nodes per move (0: use --tc)")
    r.add_argument("--tc", default="8+0.08")
    r.add_argument("--pairs", type=int, default=4, help="game pairs per iteration")
    r.add_argument("--conc", type=int, default=4)
    r.add_argument("--total-pairs", type=int, default=2000, help="N: planned game pairs (sets the schedules)")
    r.add_argument("--r-end", type=float, default=0.002)
    r.add_argument("--hours", type=float, default=0)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--params", nargs="*", help="subset of the tunables (default: all)")
    r.add_argument("--print-every", type=int, default=10)
    v = sub.add_parser("average")
    v.add_argument("states", nargs="+")
    v.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else average(a)
