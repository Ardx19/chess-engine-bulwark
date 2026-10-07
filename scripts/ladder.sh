#!/usr/bin/env bash
# Selectivity-ladder SPRTs (B.4): step k is tested with steps 1..k ON against steps 1..k-1 ON, one binary.
# usage: scripts/ladder.sh <bin> [tc] [first_step] [last_step]
set -uo pipefail
BIN=$1; TC=${2:-4+0.04}; FIRST=${3:-1}; LAST=${4:-9}
FASTCHESS=${FASTCHESS:-$HOME/tools/fastchess/fastchess}
BOOK=${BOOK:-$HOME/data/books/8moves_v3.pgn}
STEPS=(x Aspiration RFP NMP LMR KillersHistory LMPFutility IIR Improving SEEPrune)
OUT=results/ladder; mkdir -p $OUT
opts() {   # $1 = highest step enabled
  local o=""
  for i in $(seq 1 9); do
    if [ "$i" -le "$1" ]; then v=true; else v=false; fi
    o="$o option.${STEPS[$i]}=$v"
  done
  echo "$o"
}
for k in $(seq $FIRST $LAST); do
  NEWOPT=$(opts $k); BASEOPT=$(opts $((k-1)))
  echo "== step $k: ${STEPS[$k]} ==" | tee -a $OUT/summary.txt
  $FASTCHESS -engine cmd=$BIN name=new $NEWOPT -engine cmd=$BIN name=base $BASEOPT \
    -each tc=$TC option.Hash=16 option.Threads=1 option.TMScaling=false \
    -openings file=$BOOK format=pgn order=random -repeat -games 2 -rounds ${ROUNDS:-300} -concurrency 5 \
    -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
    -resign movecount=3 score=600 -draw movenumber=40 movecount=8 score=10 -maxmoves 250 \
    -pgnout file=$OUT/step$k.pgn -report penta=true > $OUT/step$k.log 2>&1
  grep -E "^(Elo|Games|SPRT|LLR)" $OUT/step$k.log | tail -4 | tee -a $OUT/summary.txt
done
