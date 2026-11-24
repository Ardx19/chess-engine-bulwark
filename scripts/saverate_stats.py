"""Save-rate statistics (B.6 protocol, step 4).

One file:  save rate (draws + wins) / games with a 95% cluster-bootstrap CI.
Two files (A = baseline, B = treatment, same position set): paired by position id
  * difference in save rate with a 95% paired bootstrap CI, resampling clusters (positions sharing a source line);
  * exact two-sided binomial test on the discordant pairs (B saved / A lost vs A saved / B lost);
  * the per-position W/D/L transition table (rows A, columns B).

Usage: python3 scripts/saverate_stats.py A.jsonl [B.jsonl] [--boot 10000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided p-value for k successes out of n at p = 0.5."""
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] * (1 + 1e-9)))


def cluster_boot(clusters: dict, stat, boot: int, seed: int = 1):
    keys = list(clusters)
    rng = random.Random(seed)
    vals = []
    for _ in range(boot):
        sample = [clusters[keys[rng.randrange(len(keys))]] for _ in keys]
        vals.append(stat([x for c in sample for x in c]))
    vals.sort()
    return vals[int(0.025 * boot)], vals[int(0.975 * boot) - 1]


def single(rows: dict, boot: int) -> dict:
    by_c = defaultdict(list)
    for r in rows.values():
        by_c[r.get("cluster", r["id"])].append(1.0 if r["saved"] else 0.0)
    xs = [x for c in by_c.values() for x in c]
    rate = sum(xs) / len(xs)
    lo, hi = cluster_boot(by_c, lambda v: sum(v) / len(v), boot)
    wdl = {k: sum(r["result"] == k for r in rows.values()) for k in "WDL"}
    reasons = defaultdict(int)
    for r in rows.values():
        reasons[r["reason"]] += 1
    return {"games": len(xs), "W": wdl["W"], "D": wdl["D"], "L": wdl["L"], "save_rate": rate, "ci95": [lo, hi],
            "clusters": len(by_c), "end_reasons": dict(reasons), "time_losses": sum(r.get("time_loss", False) for r in rows.values())}


def paired(a: dict, b: dict, boot: int) -> dict:
    ids = sorted(set(a) & set(b))
    by_c = defaultdict(list)
    for i in ids:
        by_c[a[i].get("cluster", i)].append((float(a[i]["saved"]), float(b[i]["saved"])))
    diff = lambda v: sum(y - x for x, y in v) / len(v)    # noqa: E731
    pairs = [p for c in by_c.values() for p in c]
    lo, hi = cluster_boot(by_c, diff, boot)
    b_only = sum(1 for x, y in pairs if y > x)
    a_only = sum(1 for x, y in pairs if x > y)
    table = {ra: {rb: 0 for rb in "WDL"} for ra in "WDL"}
    for i in ids:
        table[a[i]["result"]][b[i]["result"]] += 1
    return {"paired_positions": len(ids), "only_in_A": len(set(a) - set(b)), "only_in_B": len(set(b) - set(a)),
            "save_A": sum(x for x, _ in pairs) / len(pairs), "save_B": sum(y for _, y in pairs) / len(pairs),
            "diff_B_minus_A": diff(pairs), "diff_ci95": [lo, hi], "clusters": len(by_c),
            "discordant": {"B_saved_A_lost": b_only, "A_saved_B_lost": a_only,
                           "exact_p_two_sided": binom_two_sided(b_only, b_only + a_only)},
            "transition_A_rows_B_cols": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--json")
    x = ap.parse_args()
    A = load(x.a)
    rep = {"A": {"file": x.a, **single(A, x.boot)}}
    if x.b:
        B = load(x.b)
        rep["B"] = {"file": x.b, **single(B, x.boot)}
        rep["paired"] = paired(A, B, x.boot)
    print(json.dumps(rep, indent=1))
    if x.json:
        json.dump(rep, open(x.json, "w"), indent=1)


if __name__ == "__main__":
    main()
