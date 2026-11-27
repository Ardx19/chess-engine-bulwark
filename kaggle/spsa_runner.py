"""Bulwark SPSA chain for one Kaggle CPU session (no GPU): builds the tuning engine (-DBULWARK_TUNE=ON) and runs
scripts/spsa.py until HOURS is used up, saving the chain state to /kaggle/working/spsa-s<SESSION>.json after every
iteration. If that file is found among the inputs (an earlier version's output), the chain resumes from it.
Several sessions are combined on the laptop with: python3 scripts/spsa.py average results/kaggle_spsa/s*/spsa-s*.json

Input: the bulwark-match dataset (bulwark.bundle, the .nnue files, 8moves_v3.pgn, fastchess.bin).
Pushed by scripts/kaggle_match.sh with KIND=spsa, which fills in the parameters below.
"""
import glob
import os
import shutil
import subprocess

# ---- parameters (scripts/kaggle_match.sh rewrites these lines) ----
SESSION = 0
HOURS = 3.0
CONC = 4
NET = "net-a573c497.nnue"
NODES = 25000
TOTAL_PAIRS = 1500
PAIRS = 4
# --------------------------------------------------------------------

W = os.environ.get("MR_WORK", "/kaggle/working")
T = os.environ.get("MR_TMP", "/kaggle/temp_bw")
INPUT = os.environ.get("MR_INPUT", "/kaggle/input")
os.makedirs(T, exist_ok=True)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find(name):
    hits = glob.glob(f"{INPUT}/**/{name}", recursive=True)
    if not hits:
        raise SystemExit(f"{name} not found under {INPUT}")
    return hits[0]


print("CPUs:", os.cpu_count(), flush=True)
REPO = f"{T}/bulwark"
sh(f"git clone -q '{find('bulwark.bundle')}' {REPO} && cd {REPO} && git log --oneline -1")
sh(f"cd {REPO} && cmake -S . -B build-tune -DBULWARK_TUNE=ON -DCMAKE_BUILD_TYPE=Release > /dev/null "
   f"&& cmake --build build-tune -j4 --target bulwark > /dev/null")
FC = f"{T}/fastchess"
sh(f"cp '{find('fastchess.bin')}' {FC} && chmod +x {FC}")
if subprocess.run(f"{FC} -version", shell=True, capture_output=True).returncode != 0:
    sh(f"git clone -q --depth 1 https://github.com/Disservin/fastchess {T}/fc && make -C {T}/fc -j4 > /dev/null && cp {T}/fc/fastchess {FC}")
state = f"{W}/spsa-s{SESSION}.json"
prev = glob.glob(f"{INPUT}/**/spsa-s{SESSION}.json", recursive=True)
if prev and not os.path.exists(state):
    shutil.copyfile(prev[0], state)
    print("resuming from", prev[0], flush=True)
sh(f"python3 {REPO}/scripts/spsa.py run --engine {REPO}/build-tune/bulwark --state {state} --fastchess {FC} "
   f"--book '{find('8moves_v3.pgn')}' --evalfile '{find(NET)}' --nodes {NODES} --pairs {PAIRS} --conc {CONC} "
   f"--total-pairs {TOTAL_PAIRS} --hours {HOURS - 0.1} --seed {SESSION + 1}")
