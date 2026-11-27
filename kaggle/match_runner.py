"""Bulwark match runner for one Kaggle CPU session (no GPU): engine A vs engine B, same binary, only EvalFile differs.

Plays chunks of game pairs with fastchess until HOURS is used up, each chunk with its own opening seed, and writes the
summed pentanomial counts to /kaggle/working/results.json after every chunk, so a session that ends early keeps all
finished games. Several sessions (different SESSION values) are merged on the laptop with scripts/merge_penta.py.

Input: the bulwark-match dataset (bulwark.bundle, the .nnue files, 8moves_v3.pgn, fastchess.bin).
Pushed by scripts/kaggle_match.sh, which fills in the parameters below.
"""
import glob
import json
import os
import re
import subprocess
import time

# ---- parameters (scripts/kaggle_match.sh rewrites these lines) ----
SESSION = 0
HOURS = 3.0
CONC = 3
TC = "8+0.08"
NEW = "net-fcab4618.nnue"
BASE = "net-a573c497.nnue"
ROUNDS_PER_CHUNK = 9           # 18 games per chunk
FIRST_CHUNK_EST_S = 600        # time budget assumed for the first chunk
# --------------------------------------------------------------------

START = time.time()
W = os.environ.get("MR_WORK", "/kaggle/working")
T = os.environ.get("MR_TMP", "/kaggle/temp_bw")
INPUT = os.environ.get("MR_INPUT", "/kaggle/input")
os.makedirs(T, exist_ok=True)


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, text=True, **kw)


def find(name):
    hits = glob.glob(f"{INPUT}/**/{name}", recursive=True)
    if not hits:
        raise SystemExit(f"{name} not found under {INPUT}")
    return hits[0]


print("CPUs:", os.cpu_count(), flush=True)
sh(f"git clone -q '{find('bulwark.bundle')}' {T}/bulwark && cd {T}/bulwark && git log --oneline -1")
sh(f"cd {T}/bulwark && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build build -j4 --target bulwark > /dev/null")
ENGINE = f"{T}/bulwark/build/bulwark"
FC = f"{T}/fastchess"
sh(f"cp '{find('fastchess.bin')}' {FC} && chmod +x {FC}")
if subprocess.run(f"{FC} -version", shell=True, capture_output=True).returncode != 0:   # rebuild if it will not run
    sh(f"git clone -q --depth 1 https://github.com/Disservin/fastchess {T}/fc && make -C {T}/fc -j4 > /dev/null && cp {T}/fc/fastchess {FC}")
NEW_P, BASE_P, BOOK = find(NEW), find(BASE), find("8moves_v3.pgn")
nps = {}
for name, p in (("new", NEW_P), ("base", BASE_P)):
    out = subprocess.run(f"printf 'setoption name EvalFile value {p}\\nbench 12\\nquit\\n' | {ENGINE}", shell=True,
                         capture_output=True, text=True).stdout
    m = re.search(r"Nodes/second: (\d+)", out)
    nps[name] = int(m.group(1)) if m else None
print("nps", nps, flush=True)

penta = [0, 0, 0, 0, 0]
res = {"session": SESSION, "tc": TC, "conc": CONC, "new": NEW, "base": BASE, "nps": nps, "chunks": 0,
       "penta": penta, "time_losses": 0, "cpus": os.cpu_count()}
chunk_s = []
while True:
    left = HOURS * 3600 - (time.time() - START)
    est = max(chunk_s) if chunk_s else FIRST_CHUNK_EST_S
    if left < est * 1.1:
        break
    seed = SESSION * 100003 + res["chunks"] + 1
    pgn = f"{T}/chunk.pgn"
    if os.path.exists(pgn):
        os.remove(pgn)
    t0 = time.time()
    out = subprocess.run(
        [FC, "-engine", f"cmd={ENGINE}", "name=new", f"option.EvalFile={NEW_P}",
         "-engine", f"cmd={ENGINE}", "name=base", f"option.EvalFile={BASE_P}",
         "-each", f"tc={TC}", "option.Hash=16", "option.Threads=1",
         "-openings", f"file={BOOK}", "format=pgn", "order=random", "-srand", str(seed),
         "-repeat", "-games", "2", "-rounds", str(ROUNDS_PER_CHUNK), "-concurrency", str(CONC),
         "-resign", "movecount=3", "score=600", "-draw", "movenumber=40", "movecount=8", "score=10",
         "-maxmoves", "250", "-pgnout", f"file={pgn}", "-report", "penta=true"],
        capture_output=True, text=True).stdout
    chunk_s.append(time.time() - t0)
    m = re.findall(r"Ptnml\(0-2\): \[(\d+), (\d+), (\d+), (\d+), (\d+)\]", out)
    if not m:
        print(out[-2000:], flush=True)
        raise SystemExit("no pentanomial line in fastchess output")
    for i, v in enumerate(m[-1]):
        penta[i] += int(v)
    res["time_losses"] += open(pgn).read().count('Termination "time forfeit"')
    res["chunks"] += 1
    res["minutes"] = round((time.time() - START) / 60, 1)
    with open(f"{W}/results.json.tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(f"{W}/results.json.tmp", f"{W}/results.json")
    print(f"chunk {res['chunks']} ({chunk_s[-1] / 60:.1f} min): penta {penta}, pairs {sum(penta)}, "
          f"time losses {res['time_losses']}", flush=True)
print("DONE", json.dumps(res), flush=True)
