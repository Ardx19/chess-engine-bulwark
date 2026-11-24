"""Results registry: one CSV row per reported run (results/registry.csv, tracked in git).

  python3 scripts/registry.py add --kind sprt --log results/sprt-x/log.txt --new <name> --base <name> --setting "8+0.08" [--note ..]
  python3 scripts/registry.py add --kind saverate --stats results/saverate/dev-d2-vs-off.json --new d2 --base off --setting "nodes 50k vs SF 25k"
  python3 scripts/registry.py show

`add` parses a fastchess log (last Elo / LLR / Games / SPRT lines, and the W/D/L counts) or a saverate_stats JSON,
and stamps the row with the date, git commit and the engine's bench node count.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "registry.csv")
FIELDS = ["date", "commit", "bench", "kind", "new", "base", "setting", "games", "wins", "draws", "losses",
          "elo", "elo_err", "llr", "bounds", "verdict", "metric", "ci95", "source", "note"]


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:      # noqa: BLE001
        return ""


def bench(binary):
    try:
        out = subprocess.check_output([binary, "bench"], text=True, timeout=300)
        m = re.search(r"Nodes searched: (\d+)", out)
        return m.group(1) if m else ""
    except Exception:      # noqa: BLE001
        return ""


def parse_fastchess(path):
    text = open(path, errors="replace").read()
    row = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"Elo: ([-\d.]+) \+/- ([\d.]+)", line)
        if m:
            row["elo"], row["elo_err"] = m.group(1), m.group(2)
        m = re.match(r"LLR: ([-\d.]+) (?:\([-\d.]+%\) )?\([-\d.]+, [-\d.]+\) \[([-\d.]+), ([-\d.]+)\]", line)
        if m:
            row["llr"], row["bounds"] = m.group(1), f"[{m.group(2)}, {m.group(3)}]"
        m = re.match(r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)", line)
        if m:
            row["games"], row["wins"], row["losses"], row["draws"] = m.groups()
    row["verdict"] = "H1" if "H1 was accepted" in text else "H0" if "H0 was accepted" in text else "no verdict"
    return row


def parse_stats(path):
    rep = json.load(open(path))
    if "paired" in rep:
        p = rep["paired"]
        return {"games": p["paired_positions"], "metric": f"save {p['save_A']:.3f} -> {p['save_B']:.3f} (diff {p['diff_B_minus_A']:+.3f}, p={p['discordant']['exact_p_two_sided']:.3g})",
                "ci95": f"[{p['diff_ci95'][0]:+.3f}, {p['diff_ci95'][1]:+.3f}]"}
    a = rep["A"]
    return {"games": a["games"], "wins": a["W"], "draws": a["D"], "losses": a["L"], "metric": f"save {a['save_rate']:.3f}",
            "ci95": f"[{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("--kind", required=True, help="sprt | netcheck | saverate | fortress | tbsuite | strength | other")
    add.add_argument("--log", help="fastchess log")
    add.add_argument("--stats", help="saverate_stats.py JSON")
    add.add_argument("--new", default="")
    add.add_argument("--base", default="")
    add.add_argument("--setting", default="")
    add.add_argument("--metric", default="")
    add.add_argument("--note", default="")
    add.add_argument("--binary", default=os.path.join(ROOT, "build", "bulwark"))
    add.add_argument("--commit", help="override (historical runs)")
    add.add_argument("--bench", help="override (historical runs)")
    add.add_argument("--date", help="override (historical runs)")
    sub.add_parser("show")
    a = ap.parse_args()
    if a.cmd == "show":
        if os.path.exists(PATH):
            print(open(PATH).read())
        return
    row = {k: "" for k in FIELDS}
    row.update(date=a.date or dt.date.today().isoformat(), commit=a.commit or git_commit(),
               bench=a.bench or bench(a.binary), kind=a.kind, new=a.new,
               base=a.base, setting=a.setting, metric=a.metric.strip(), note=a.note, source=a.log or a.stats or "")
    if a.log:
        row.update(parse_fastchess(a.log))
    if a.stats:
        row.update(parse_stats(a.stats))
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    new_file = not os.path.exists(PATH)
    with open(PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
