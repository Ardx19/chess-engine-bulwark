"""Summarise a scripts/overnight.sh run directory as plain text.

usage: python3 scripts/overnight_report.py <run_dir>
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

OK_TERMINATIONS = {"normal", "adjudication"}
FEATURES = ["TMScaling", "D3_Rule50Scale", "D5_StalemateGuard", "D4_Syzygy", "IIR"]


def games(pgn):
    """Header tags of each game in a PGN file (empty if the file is missing)."""
    if not pgn.exists():
        return []
    out, tags = [], None
    for line in pgn.read_text(errors="replace").splitlines():
        m = re.match(r'\[(\w+) "(.*)"\]', line)
        if not m:
            continue
        if m.group(1) == "Event":
            tags = {}
            out.append(tags)
        if tags is not None:
            tags[m.group(1)] = m.group(2)
    return out


def last_stats(log):
    """Last Elo / Games / LLR / SPRT lines of a fastchess log."""
    if not log.exists():
        return ["(no log: stage did not run)"]
    found = {}
    for line in log.read_text(errors="replace").splitlines():
        for p in ("Elo:", "Games:", "LLR:", "SPRT"):
            if line.startswith(p):
                found[p] = line.strip()
    return list(found.values()) or ["(no results in log)"]


def verdict(lines):
    text = " ".join(lines)
    if "H1 was accepted" in text:
        return "PASS (H1 accepted)"
    if "H0 was accepted" in text:
        return "FAIL (H0 accepted)"
    return "no verdict (hit the game cap; use the Elo estimate)"


def problems(gs):
    """Count of games that ended in anything other than a normal finish or adjudication."""
    return Counter(g.get("Termination", "?") for g in gs if g.get("Termination", "?") not in OK_TERMINATIONS)


def legality_block(name, pgn):
    gs = games(pgn)
    bad = problems(gs)
    status = "CLEAN" if gs and not bad else ("NO GAMES" if not gs else "PROBLEMS: " + dict(bad).__repr__())
    results = Counter(g.get("Result", "?") for g in gs)
    return [f"{name}: {len(gs)} games, {status}", f"  results {dict(results)}"]


def expected(diff):
    return 1 / (1 + 10 ** (-diff / 400))


def sf_block(pgn):
    rows, per_level = [], []
    gs = games(pgn)
    levels = sorted({int(n[2:]) for g in gs for n in (g.get("White", ""), g.get("Black", "")) if re.fullmatch(r"SF\d+", n)})
    for level in levels:
        name = f"SF{level}"
        w = d = l = 0
        for g in gs:
            if {g.get("White"), g.get("Black")} != {"bulwark", name}:
                continue
            r = g.get("Result")
            if r == "1/2-1/2":
                d += 1
            elif r in ("1-0", "0-1"):
                won = (r == "1-0") == (g.get("White") == "bulwark")
                w, l = (w + 1, l) if won else (w, l + 1)
        n = w + d + l
        if not n:
            rows.append(f"  vs {name}: no games")
            continue
        s = (w + d / 2) / n
        diff = f"{-400 * math.log10(1 / s - 1):+.0f}" if 0 < s < 1 else ("+inf" if s == 1 else "-inf")
        rows.append(f"  vs {name}: +{w} ={d} -{l} ({n} games), score {s:.1%}, Elo diff {diff}")
        per_level.append((level, n, w + d / 2))
    total = sum(n for _, n, _ in per_level)
    points = sum(p for _, _, p in per_level)
    if per_level and 0 < points < total:
        lo, hi = min(levels) - 1500, max(levels) + 1500
        for _ in range(100):   # score is monotone in rating: bisect for the maximum-likelihood rating
            mid = (lo + hi) / 2
            if sum(p - n * expected(mid - lv) for lv, n, p in per_level) > 0:
                lo = mid
            else:
                hi = mid
        info = sum(n * expected(lo - lv) * (1 - expected(lo - lv)) for lv, n, _ in per_level) * (math.log(10) / 400) ** 2
        rows.append(f"  rough rating estimate (CCRL 40/4 scale via UCI_Elo): {lo:.0f} +/- {1.96 / math.sqrt(info):.0f}")
    bad = problems(gs)
    rows.append(f"  {len(gs)} games, " + ("no illegal moves / time losses" if not bad else f"PROBLEMS: {dict(bad)}"))
    return rows


def main():
    run = Path(sys.argv[1])
    out = [f"Overnight run {run.name}"]
    meta = run / "meta.txt"
    out += meta.read_text().splitlines() if meta.exists() else []
    if (run / "progress.txt").exists():
        out += ["", "Timeline:"] + ["  " + l for l in (run / "progress.txt").read_text().splitlines()]

    out += ["", "1-2. One-switch SPRTs (new = switch on, base = switch off)"]
    for f in FEATURES:
        lines = last_stats(run / "features" / f"{f}.log")
        bad = problems(games(run / "features" / f"{f}.pgn"))
        out.append(f"  {f}: {verdict(lines)}" + (f"  PROBLEMS: {dict(bad)}" if bad else ""))
        out += ["      " + l for l in lines]

    out += ["", "3. Final build (ladder settings) vs Phase 2 build"]
    out += ["      " + l for l in last_stats(run / "vs_p2.log")]
    bad = problems(games(run / "vs_p2.pgn"))
    if bad:
        out.append(f"      PROBLEMS: {dict(bad)}")

    out += ["", "4-5. Legality runs"]
    out += ["  " + l for l in legality_block("soak 8+0.08", run / "soak.pgn")]
    out += ["  " + l for l in legality_block("Chess960/DFRC", run / "dfrc.pgn")]

    for pgn in sorted(run.glob("sf*.pgn")):
        out += ["", f"6. Stockfish 19 UCI_Elo gauntlet (60+0.6), {pgn.stem}"]
        out += sf_block(pgn)
    print("\n".join(out))


if __name__ == "__main__":
    main()
