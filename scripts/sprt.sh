#!/usr/bin/env bash
# Timed SPRT per B.6 (STC): scripts/sprt.sh <new_bin> <base_bin> [bounds_lo bounds_hi] [tc] [concurrency]
# Laptop discipline: plugged in, nothing else heavy running; 5 concurrent games.
set -euo pipefail
NEW=$1; BASE=$2; LO=${3:-0}; HI=${4:-10}; TC=${5:-8+0.08}; CONC=${6:-5}
FASTCHESS=${FASTCHESS:-$HOME/tools/fastchess/fastchess}
BOOK=${BOOK:-$HOME/data/books/8moves_v3.pgn}
OUT=results/sprt-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
"$FASTCHESS" \
  -engine cmd="$NEW" name=new -engine cmd="$BASE" name=base \
  -each tc="$TC" option.Hash=16 option.Threads=1 \
  -openings file="$BOOK" format=pgn order=random -repeat \
  -games 2 -rounds 10000 -concurrency "$CONC" \
  -sprt elo0="$LO" elo1="$HI" alpha=0.05 beta=0.05 \
  -resign movecount=3 score=600 -draw movenumber=40 movecount=8 score=10 \
  -maxmoves 200 -pgnout file="$OUT/games.pgn" \
  -report penta=true 2>&1 | tee "$OUT/log.txt"
