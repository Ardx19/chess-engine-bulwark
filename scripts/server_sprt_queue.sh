#!/usr/bin/env bash
# One-switch SPRT queue for a shared many-core server, everything inside this project folder, lowest priority.
#
#   scripts/server_sprt_queue.sh SingularExt CorrHist Razoring ...     # switches to test, one after another
#   CONC=16 TC=4+0.04 ROUNDS=1000 scripts/server_sprt_queue.sh ...     # defaults shown
#
# For each switch: new = engine with the switch on, base = switch off, everything else at its default; fastchess
# SPRT [ELO0, ELO1] (default [0, 5]) on the 8moves_v3 book, up to 2*ROUNDS games. Builds fastchess (tools/fastchess)
# and the engine (build/) if missing, downloads the book (data/books). Results: results/server_sprt/<switch>.log and
# summary.txt (Elo, games, LLR, verdict, time losses). Rerunning skips switches that already have a summary entry.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/local_env.sh
CONC=${CONC:-16}; TC=${TC:-4+0.04}; ROUNDS=${ROUNDS:-1000}; ELO0=${ELO0:-0}; ELO1=${ELO1:-5}
OUT=results/server_sprt; mkdir -p "$OUT" tools data/books
LOW=(nice -n 19); command -v ionice >/dev/null 2>&1 && LOW+=(ionice -c 3)
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/queue.log"; }

if [ ! -x tools/fastchess/fastchess ]; then
  log "building fastchess"
  rm -rf tools/fastchess && git clone -q --depth 1 https://github.com/Disservin/fastchess tools/fastchess
  (cd tools/fastchess && "${LOW[@]}" make -j8 >/dev/null)
fi
if [ ! -f data/books/8moves_v3.pgn ]; then
  log "downloading 8moves_v3 book"
  curl -sfL -o data/books/8moves_v3.pgn.zip https://github.com/official-stockfish/books/raw/master/8moves_v3.pgn.zip
  python3 -c "import zipfile; zipfile.ZipFile('data/books/8moves_v3.pgn.zip').extractall('data/books')"
fi
log "building engine at $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && "${LOW[@]}" cmake --build build -j8 >/dev/null
cp build/bulwark "$OUT/bulwark"
BENCH=$("$OUT/bulwark" bench | awk '/Nodes searched/{print $3}')
log "engine bench $BENCH; switches: $*; conc $CONC, tc $TC, bounds [$ELO0, $ELO1], max $((2 * ROUNDS)) games"

for OPT in "$@"; do
  if grep -q "^== $OPT " "$OUT/summary.txt" 2>/dev/null; then log "skip $OPT (done)"; continue; fi
  log "start $OPT"
  "${LOW[@]}" tools/fastchess/fastchess \
    -engine cmd="$OUT/bulwark" name=new "option.$OPT=true" -engine cmd="$OUT/bulwark" name=base "option.$OPT=false" \
    -each tc="$TC" option.Hash=16 option.Threads=1 \
    -openings file=data/books/8moves_v3.pgn format=pgn order=random -repeat -games 2 -rounds "$ROUNDS" -concurrency "$CONC" \
    -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.05 \
    -resign movecount=3 score=600 -draw movenumber=40 movecount=8 score=10 -maxmoves 250 \
    -pgnout file="$OUT/$OPT.pgn" -report penta=true > "$OUT/$OPT.log" 2>&1 || true
  tl=$(grep -c 'Termination "time forfeit"' "$OUT/$OPT.pgn" 2>/dev/null || true)
  { echo "== $OPT [$ELO0,$ELO1] tc=$TC conc=$CONC bench=$BENCH"
    grep -E "^(Elo|Games|LLR|SPRT)" "$OUT/$OPT.log" | tail -4
    echo "time losses: ${tl:-0}"; } >> "$OUT/summary.txt"
  log "done $OPT"
done
log "queue finished"; cat "$OUT/summary.txt"
