#!/usr/bin/env bash
# Strength benchmark vs Stockfish UCI_Elo settings (B.6): scripts/strength.sh <bin> "<levels>" [rounds] [engine options...]
# 60+0.6, 8moves_v3, gauntlet; levels = estimate -200 / +0 / +200 (e.g. "2550 2750 2950"). Overnight, laptop idle.
# Extra arguments are passed to the engine as option.X=Y (e.g. option.EvalFile=nets/n1.nnue).
set -euo pipefail
BIN=$1; LEVELS=${2:-2550 2750 2950}; ROUNDS=${3:-100}; shift 3 || true
FASTCHESS=${FASTCHESS:-$HOME/tools/fastchess/fastchess}
SF=${SF:-$HOME/tools/stockfish/stockfish/stockfish-linux-x86-64-universal}
BOOK=${BOOK:-$HOME/data/books/8moves_v3.pgn}
TB=${SYZYGY_PATH:-$HOME/data/syzygy/tb345}
OUT=results/strength-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
SFE=(); for e in $LEVELS; do SFE+=(-engine cmd="$SF" name=SF$e option.UCI_LimitStrength=true option.UCI_Elo=$e); done
"$FASTCHESS" -tournament gauntlet -seeds 1 -engine cmd="$BIN" name=bulwark option.SyzygyPath="$TB" "$@" "${SFE[@]}" \
  -each tc=60+0.6 option.Hash=16 option.Threads=1 \
  -openings file="$BOOK" format=pgn order=random -repeat -games 2 -rounds "$ROUNDS" -concurrency ${CONC:-5} \
  -resign movecount=3 score=600 twosided=true -draw movenumber=40 movecount=8 score=10 -maxmoves 250 \
  -pgnout file="$OUT/games.pgn" 2>&1 | tee "$OUT/log.txt"
python3 -c "import sys; sys.path.insert(0, 'scripts'); from overnight_report import sf_block; from pathlib import Path; print('\n'.join(sf_block(Path('$OUT/games.pgn'))))" | tee "$OUT/REPORT.txt"
python3 scripts/registry.py add --kind strength --new "$(basename "$BIN") $*" --base "SF UCI_Elo $LEVELS" --setting "60+0.6" \
  --metric "$(grep -o 'rough rating estimate.*' "$OUT/REPORT.txt" || true)" --note "$OUT"
