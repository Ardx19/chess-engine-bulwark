#!/usr/bin/env bash
# One-switch SPRT: <bin> <OptionName> [elo0] [elo1] [tc]. new = option true, base = option false, all else default.
# Used for the D-features (D3/D4/D5) and non-regression checks. Syzygy path is set for both sides.
set -uo pipefail
BIN=$1; OPT=$2; LO=${3:--5}; HI=${4:-5}; TC=${5:-4+0.04}
FASTCHESS=${FASTCHESS:-$HOME/tools/fastchess/fastchess}
BOOK=${BOOK:-$HOME/data/books/8moves_v3.pgn}
TB=${SYZYGY_PATH:-$HOME/data/syzygy/tb345}
OUT=results/features; mkdir -p $OUT
$FASTCHESS -engine cmd=$BIN name=new option.$OPT=true -engine cmd=$BIN name=base option.$OPT=false \
  -each tc=$TC option.Hash=16 option.Threads=1 option.SyzygyPath=$TB \
  -openings file=$BOOK format=pgn order=random -repeat -games 2 -rounds ${ROUNDS:-300} -concurrency 5 \
  -sprt elo0=$LO elo1=$HI alpha=0.05 beta=0.05 \
  -resign movecount=3 score=600 -draw movenumber=40 movecount=8 score=10 -maxmoves 250 \
  -pgnout file=$OUT/$OPT.pgn -report penta=true > $OUT/$OPT.log 2>&1
{ echo "== $OPT [$LO,$HI] tc=$TC =="; grep -E "^(Elo|Games|LLR|SPRT)" $OUT/$OPT.log | tail -4; } | tee -a $OUT/summary.txt
