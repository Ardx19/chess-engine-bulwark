#!/usr/bin/env bash
# Phase 4 data pilot / production round for ONE Kaggle CPU session (notebook cell: !bash kaggle/pilot.sh).
# Env: SESSION (shard id, default 0), NUM_SESSIONS (default 5), MAX_TARS (pilot: 3; production: large),
#      RUN (Lc0 run dir, default test91), SF_NODES (default 2000), RELABEL_LIMIT (pilot: 20000; production: 0 = all)
# Requires internet. Everything stays inside the project folder: outputs in OUT (default data/run), Stockfish in
# tools/sf, tablebases in tools/syzygy. Set NUM_SESSIONS/SESSION to split the tars into chunks;
# WORKERS caps the Stockfish processes (default: all cores), STOP_H the hours after which no new tar is taken,
# MAX_KEPT the number of kept positions after which no new tar is taken (0 = no limit).
set -euo pipefail
SESSION=${SESSION:-0}; NUM=${NUM_SESSIONS:-5}; MAX_TARS=${MAX_TARS:-3}; RUN=${RUN:-test91}
SF_NODES=${SF_NODES:-2000}; RELABEL_LIMIT=${RELABEL_LIMIT:-20000}
OUT=${OUT:-data/run}            # relative to the project folder (on Kaggle: OUT=/kaggle/working/data)
cd "$(dirname "$0")/.."

mkdir -p "$OUT"
echo "== machine"; nproc; free -g | head -2; df -h "$OUT" | tail -1

echo "== build converter"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENGINE_ARCH=x86-64-v3 >/dev/null
cmake --build build -j"$(nproc)" --target lc0_convert

echo "== stockfish (pinned release; record the hash)"
SF_URL=https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz
[ -x tools/sf/stockfish/stockfish-linux-x86-64-universal ] || { mkdir -p tools/sf && curl -sL "$SF_URL" | tar xz -C tools/sf; }
SF=$PWD/tools/sf/stockfish/stockfish-linux-x86-64-universal
sha256sum "$SF"

echo "== syzygy 3-5 piece (tablebase label override)"
TB=$PWD/tools/syzygy
if [ "$(ls "$TB"/*.rtbw 2>/dev/null | wc -l)" -lt 100 ]; then
  mkdir -p "$TB"
  for u in 3-4-5-wdl 3-4-5-dtz; do
    curl -sL "https://tablebase.lichess.ovh/tables/standard/$u/" | grep -o 'href="[^"]*rtb[wz]"' | sed 's/href="//;s/"//' |
      sed "s#^#https://tablebase.lichess.ovh/tables/standard/$u/#" | xargs -P 4 -n 1 curl -sfL -O --output-dir "$TB"
  done
  n=$(ls "$TB"/*.rtbw 2>/dev/null | wc -l); echo "syzygy: $n WDL files"; [ "$n" -ge 100 ] || { echo "tablebase download failed"; exit 1; }
fi

echo "== data round"
python3 -m pipeline.run --run "$RUN" --session "$SESSION" --num-sessions "$NUM" --max-tars "$MAX_TARS" --skip-tars 3 \
  --out "$OUT" --converter build/lc0_convert --stockfish "$SF" --sf-nodes "$SF_NODES" --workers "${WORKERS:-$(nproc)}" \
  --syzygy "$TB" --relabel-limit "$RELABEL_LIMIT" --stop-taking-h "${STOP_H:-10.5}" --max-kept "${MAX_KEPT:-0}"
echo "== done"; ls -la "$OUT"
