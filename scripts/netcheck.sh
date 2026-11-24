#!/usr/bin/env bash
# Net-vs-net check at fixed nodes (B.6): scripts/netcheck.sh <new.nnue> <base.nnue> [nodes] [elo0] [elo1] [concurrency]
# Same binary, only EvalFile differs. Openings: the UHO pool left over after carving the benchmark sets.
# Fixed-node games are CPU-speed independent, so up to ~10 concurrent games are fine (also on a server / Kaggle CPU).
set -euo pipefail
NEW=$1; BASE=$2; NODES=${3:-25000}; LO=${4:-0}; HI=${5:-5}; CONC=${6:-10}
BIN=${BIN:-build/bulwark}
FASTCHESS=${FASTCHESS:-$HOME/tools/fastchess/fastchess}
BOOK=${BOOK:-benchsets/pool.epd}
TB=${SYZYGY_PATH:-$HOME/data/syzygy/tb345}
OUT=results/netcheck-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
"$FASTCHESS" -engine cmd="$BIN" name=new option.EvalFile="$NEW" -engine cmd="$BIN" name=base option.EvalFile="$BASE" \
  -each nodes="$NODES" tc=inf option.Hash=16 option.Threads=1 option.SyzygyPath="$TB" \
  -openings file="$BOOK" format=epd order=random -repeat -games 2 -rounds ${ROUNDS:-10000} -concurrency "$CONC" \
  -sprt elo0="$LO" elo1="$HI" alpha=0.05 beta=0.05 \
  -resign movecount=3 score=600 -draw movenumber=40 movecount=8 score=10 -maxmoves 250 \
  -pgnout file="$OUT/games.pgn" -report penta=true 2>&1 | tee "$OUT/log.txt"
python3 scripts/registry.py add --kind netcheck --log "$OUT/log.txt" --new "$(basename "$NEW")" --base "$(basename "$BASE")" --setting "nodes=$NODES"
