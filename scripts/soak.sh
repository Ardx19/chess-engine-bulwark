#!/usr/bin/env bash
# Legality / time-loss soak: self-play with illegal-move and timeout detection. scripts/soak.sh <bin> [games] [tc]
set -euo pipefail
BIN=$1; GAMES=${2:-400}; TC=${3:-2+0.02}
FASTCHESS=${FASTCHESS:-$HOME/tools/fastchess/fastchess}
BOOK=${BOOK:-$HOME/data/books/8moves_v3.pgn}
mkdir -p results
"$FASTCHESS" -engine cmd="$BIN" name=a -engine cmd="$BIN" name=b \
  -each tc="$TC" option.Hash=16 -openings file="$BOOK" format=pgn order=random \
  -games 2 -rounds $((GAMES/2)) -concurrency 5 -maxmoves 300 \
  -pgnout file=results/soak.pgn 2>&1 | tee results/soak.log
