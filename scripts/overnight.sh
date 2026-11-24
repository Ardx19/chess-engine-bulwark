#!/usr/bin/env bash
# Unattended WP-A close-out queue (laptop plugged in and idle). Stages, in order:
#   1. one-switch SPRTs [-5,5] at 4+0.04, 600-game cap: TMScaling, D3_Rule50Scale, D5_StalemateGuard, D4_Syzygy
#   2. IIR re-test [0,10], 1200-game cap
#   3. final build vs Phase 2 build, 400 games at 4+0.04, ladder settings (TMScaling off, no tablebases)
#   4. legality / time-loss soak: 1000 games at 8+0.08, tablebases on
#   5. Chess960/DFRC legality run: 400 games at 4+0.04 from random DFRC starts (scripts/gen_dfrc.py)
#   6. Stockfish 19 UCI_Elo gauntlet at 60+0.6 until the deadline (SF_LEVELS, default "2000 2300 2600";
#      SF_STAGE names the stage, its checkpoint and its sf*.log/.pgn, default sf)
# RESET_DEADLINE=1 replaces the stored deadline with the one given on the command line.
# then writes REPORT.txt in the run directory. The engine binary is copied first, so rebuilding during the run is safe.
#
# Checkpoints: each stage runs fastchess inside <run>/ckpt/<stage>/, where fastchess saves its state (config.json)
# after every game and on SIGINT. A finished stage gets a DONE marker. Re-running with RUN=<run dir> skips finished
# stages and resumes an interrupted one from its config.json; the deadline stays the one set on the first start.
# A STOP file in the run dir (set by scripts/watchdog.sh) makes the queue exit before the next stage.
# Exit codes: 0 finished, 2 a stage was interrupted or crashed, 3 stopped by STOP.
# usage: [RUN=<run dir>] scripts/overnight.sh [deadline HH:MM, default 09:45]; SMOKE=1 runs a few fast games per stage.
set -uo pipefail
cd "$(dirname "$0")/.."
DEADLINE=${1:-09:45}; SMOKE=${SMOKE:-0}
FASTCHESS=${FASTCHESS:-$HOME/tools/fastchess/fastchess}
SF=${SF:-$HOME/tools/stockfish/stockfish/stockfish-linux-x86-64-universal}
BOOK=${BOOK:-$HOME/data/books/8moves_v3.pgn}
DFRC=${DFRC:-$HOME/data/books/dfrc_1000.epd}
TB=${SYZYGY_PATH:-$HOME/data/syzygy/tb345}
P2=$PWD/engines/bulwark-p2
RUN=$(realpath -m "${RUN:-results/overnight-$(date +%Y%m%d-%H%M)$([ "$SMOKE" = 1 ] && echo -smoke)}")
mkdir -p "$RUN/features" "$RUN/ckpt"
BIN=$RUN/bulwark; [ -x "$BIN" ] || cp build/bulwark "$BIN"
rm -f "$RUN/STOP"; echo $$ > "$RUN/queue.pid"

# Hold Windows awake (system and display) while the queue runs: "sleep: Never" alone does not stop Modern Standby,
# which froze WSL for 4 h on 2026-10-20. The request lasts while the PowerShell process lives; killing it from WSL
# does not end the Windows process, so it exits on its own within 30 s once the queue removes <run>/awake.
if command -v powershell.exe > /dev/null; then
  touch "$RUN/awake"; trap 'rm -f "$RUN/awake"' EXIT
  powershell.exe -NoProfile -Command "Add-Type -Name P -Namespace W -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);'; while (Test-Path '$(wslpath -w "$RUN/awake")') { [W.P]::SetThreadExecutionState([uint32]2147483651) | Out-Null; Start-Sleep 30 }" > /dev/null 2>&1 &
fi

if [ "$SMOKE" = 1 ]; then
  STC=1+0.01; LTC=1+0.01; SFTC=2+0.02; FR=2; IR=2; PR=2; SOAK=4; FRCR=4; SFR=1
else
  STC=4+0.04; LTC=8+0.08; SFTC=60+0.6; FR=300; IR=600; PR=200; SOAK=500; FRCR=200; SFR=100
fi
COMMON=(-concurrency 5 -autosaveinterval 1 -report penta=true)
ADJ=(-resign movecount=3 score=600 -draw movenumber=40 movecount=8 score=10 -maxmoves 250)
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/progress.txt"; }

if [ ! -s "$RUN/deadline" ] || [ "${RESET_DEADLINE:-0}" = 1 ]; then
  end=$(date -d "$DEADLINE" +%s); [ "$end" -le "$(date +%s)" ] && end=$((end + 86400))
  [ "$SMOKE" = 1 ] && end=$(($(date +%s) + 900))
  echo "$end" > "$RUN/deadline"
fi
end=$(cat "$RUN/deadline")
[ -s "$RUN/meta.txt" ] || {
  echo "commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src || echo ' (src modified)')"
  echo "started $(date '+%Y-%m-%d %H:%M'), deadline $(date -d "@$end" '+%H:%M')"
  echo "Bench: $("$BIN" bench 2>&1 | awk '/Nodes searched/ {print $3}')"; } > "$RUN/meta.txt"
log "queue started (pid $$), run dir $RUN"

# stage <name> <log> <fastchess args...>: run one stage, or resume it from its checkpoint.
# STAGE_TIMEOUT (seconds) caps the wall time; a stage stopped by that cap counts as finished.
stage() {
  local name=$1 log=$2; shift 2
  local dir=$RUN/ckpt/$name off=0 t=() rc out
  mkdir -p "$dir"
  [ -e "$dir/DONE" ] && { log "skip $name (done)"; return; }
  [ -e "$RUN/STOP" ] && { log "STOP file present, exiting before $name"; exit 3; }
  [ -n "${STAGE_TIMEOUT:-}" ] && t=(timeout -s INT -k 120 "$STAGE_TIMEOUT")
  if [ -s "$dir/config.json" ]; then
    log "resume $name from checkpoint"; off=$(stat -c %s "$log" 2>/dev/null || echo 0)
    (cd "$dir" && "${t[@]}" "$FASTCHESS" -config file=config.json >> "$log" 2>&1)
  else
    log "start $name"
    (cd "$dir" && "${t[@]}" "$FASTCHESS" "$@" > "$log" 2>&1)
  fi
  rc=$?; out=$(tail -c +$((off + 1)) "$log")
  if { [ $rc -eq 124 ] && [ -n "${STAGE_TIMEOUT:-}" ]; } || { grep -q "Finished match" <<<"$out" && ! grep -q "interrupted" <<<"$out"; }; then
    touch "$dir/DONE"; log "done $name"
  else
    log "$name stopped early (exit $rc); resume with: RUN=$RUN scripts/overnight.sh"; exit 2
  fi
}

for opt in TMScaling D3_Rule50Scale D5_StalemateGuard D4_Syzygy IIR; do
  lo=-5; hi=5; rounds=$FR
  [ $opt = IIR ] && { lo=0; hi=10; rounds=$IR; }
  stage $opt "$RUN/features/$opt.log" \
    -engine cmd=$BIN name=new option.$opt=true -engine cmd=$BIN name=base option.$opt=false \
    -each tc=$STC option.Hash=16 option.Threads=1 option.SyzygyPath=$TB \
    -openings file=$BOOK format=pgn order=random -repeat -games 2 -rounds $rounds "${COMMON[@]}" \
    -sprt elo0=$lo elo1=$hi alpha=0.05 beta=0.05 "${ADJ[@]}" -pgnout file=$RUN/features/$opt.pgn
done

stage vs_p2 "$RUN/vs_p2.log" \
  -engine cmd=$BIN name=final option.TMScaling=false -engine cmd=$P2 name=p2 \
  -each tc=$STC option.Hash=16 option.Threads=1 \
  -openings file=$BOOK format=pgn order=random -repeat -games 2 -rounds $PR "${COMMON[@]}" "${ADJ[@]}" \
  -pgnout file=$RUN/vs_p2.pgn

stage soak "$RUN/soak.log" \
  -engine cmd=$BIN name=a -engine cmd=$BIN name=b -each tc=$LTC option.Hash=16 option.SyzygyPath=$TB \
  -openings file=$BOOK format=pgn order=random -games 2 -rounds $SOAK "${COMMON[@]}" -maxmoves 300 \
  -pgnout file=$RUN/soak.pgn

stage dfrc "$RUN/dfrc.log" -variant fischerandom \
  -engine cmd=$BIN name=a -engine cmd=$BIN name=b -each tc=$STC option.Hash=16 option.SyzygyPath=$TB \
  -openings file=$DFRC format=epd order=random -repeat -games 2 -rounds $FRCR "${COMMON[@]}" -maxmoves 300 \
  -pgnout file=$RUN/dfrc.pgn

SF_LEVELS=${SF_LEVELS:-2000 2300 2600}; SF_STAGE=${SF_STAGE:-sf}
left=$((end - $(date +%s)))
if [ -e "$RUN/ckpt/$SF_STAGE/DONE" ] || [ $left -gt 600 ]; then
  SFE=(); for e in $SF_LEVELS; do SFE+=(-engine cmd=$SF name=SF$e option.UCI_LimitStrength=true option.UCI_Elo=$e); done
  STAGE_TIMEOUT=$left stage $SF_STAGE "$RUN/$SF_STAGE.log" -tournament gauntlet -seeds 1 \
    -engine cmd=$BIN name=bulwark option.SyzygyPath=$TB "${SFE[@]}" \
    -each tc=$SFTC option.Hash=16 option.Threads=1 \
    -openings file=$BOOK format=pgn order=random -repeat -games 2 -rounds $SFR "${COMMON[@]}" \
    -resign movecount=3 score=600 twosided=true -draw movenumber=40 movecount=8 score=10 -maxmoves 250 \
    -pgnout file=$RUN/$SF_STAGE.pgn
else
  log "skipping Stockfish gauntlet (under 10 min left before the deadline)"
fi

python3 scripts/overnight_report.py "$RUN" > "$RUN/REPORT.txt"
log "done: $RUN/REPORT.txt"
