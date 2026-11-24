#!/usr/bin/env bash
# Watchdog for a scripts/overnight.sh run. Every INTERVAL seconds it logs CPU temperature and clock (read from Windows
# through powershell.exe), free memory and disk to <run>/watchdog.txt, and checks the run for stalls, bad game endings
# and fastchess warnings. It exits on the first problem with a one-line reason, so the background task that started
# it wakes whoever is watching. Critical heat or memory first stops the run cleanly: a STOP file (the queue exits
# before the next stage) and SIGINT to the run's fastchess (which saves its checkpoint).
# usage: scripts/watchdog.sh <run_dir>; thresholds can be overridden through the environment.
set -uo pipefail
RUN=$(realpath "$1")
INTERVAL=${INTERVAL:-60}
WARN_C=${WARN_C:-95}; CRIT_C=${CRIT_C:-100}             # hottest Windows thermal zone, deg C
MIN_PERF=${MIN_PERF:-60}                                 # CPU clock as % of base; 5 samples below = throttling
WARN_MEM=${WARN_MEM:-1500}; CRIT_MEM=${CRIT_MEM:-700}    # MB available inside WSL
MIN_DISK=${MIN_DISK:-5}                                  # GB free on Windows C:, which holds the WSL virtual disk
STALL=${STALL:-12}                                       # minutes without any write to the run's logs
OUT=$RUN/watchdog.txt
HARMLESS="Bestmove does not match beginning of last PV"  # Stockfish with UCI_LimitStrength plays off its PV
PS='$t = (Get-Counter "\Thermal Zone Information(*)\Temperature").CounterSamples | Measure-Object CookedValue -Maximum;
$p = (Get-Counter "\Processor Information(_Total)\% Processor Performance").CounterSamples.CookedValue;
"{0} {1} {2}" -f [math]::Round($t.Maximum - 273.15), [math]::Round($p), [math]::Floor((Get-PSDrive C).Free / 1GB)'

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
fail() { say "WATCHDOG: $*"; exit 1; }
pgns() { cat "$RUN"/*.pgn "$RUN"/features/*.pgn 2>/dev/null; }
bad_endings() { pgns | grep -o '^\[Termination "[^"]*"\]' | grep -v -E '"(normal|adjudication)"'; }
warnings() { cat "$RUN"/*.log "$RUN"/features/*.log 2>/dev/null \
  | grep -i -E "warn|error|illegal|crash|disconnect|respond|stall" | grep -v "$HARMLESS"; }
stop_run() {
  touch "$RUN/STOP"
  for p in $(pgrep -x fastchess); do
    [[ $(readlink "/proc/$p/cwd") == "$RUN"/* ]] && kill -INT "$p"
  done
}

base_bad=$(bad_endings | wc -l); base_warn=$(warnings | wc -l)
hot=0; crit=0; slow=0; nosensor=0
say "watching $RUN: heat warn ${WARN_C}C / stop ${CRIT_C}C, memory warn ${WARN_MEM} / stop ${CRIT_MEM} MB, stall ${STALL} min"
while true; do
  pid=$(cat "$RUN/queue.pid" 2>/dev/null)
  if ! kill -0 "$pid" 2>/dev/null; then
    [ -s "$RUN/REPORT.txt" ] && { say "queue finished, REPORT.txt written"; exit 0; }
    [ -e "$RUN/STOP" ] && fail "queue exited after a STOP; resume with RUN=$RUN scripts/overnight.sh"
    fail "queue process ${pid:-?} is gone and there is no REPORT.txt; tail of progress: $(tail -1 "$RUN/progress.txt")"
  fi
  read -r temp perf disk < <(timeout 30 powershell.exe -NoProfile -Command "$PS" 2>/dev/null | tr -d '\r')
  [[ ${temp:-} =~ ^[0-9]+$ && ${perf:-} =~ ^[0-9]+$ ]] || { temp=""; perf=""; }
  [[ ${disk:-} =~ ^[0-9]+$ ]] || disk=$(df -BG --output=avail "$RUN" | tail -1 | tr -dc 0-9)
  mem=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
  newest=$(ls -t "$RUN"/*.log "$RUN"/features/*.log "$RUN"/progress.txt 2>/dev/null | head -1)
  idle=$(( ($(date +%s) - $(stat -c %Y "$newest")) / 60 ))
  say "temp ${temp:-?}C clock ${perf:-?}% mem ${mem}MB disk ${disk}G load $(cut -d' ' -f1 /proc/loadavg)" \
      "games $(pgns | grep -c '^\[Event ') stage $(tail -1 "$RUN/progress.txt" | cut -d' ' -f2-)"

  if [ -z "$temp" ]; then
    nosensor=$((nosensor + 1)); [ $nosensor -ge 10 ] && fail "cannot read CPU temperature from Windows (10 tries)"
  else
    nosensor=0
    [ "$temp" -ge "$CRIT_C" ] && crit=$((crit + 1)) || crit=0
    [ "$temp" -ge "$WARN_C" ] && hot=$((hot + 1)) || hot=0
    [ "$perf" -lt "$MIN_PERF" ] && slow=$((slow + 1)) || slow=0
    [ $crit -ge 2 ] && { stop_run; fail "CPU at ${temp}C twice in a row: stopped the run cleanly, checkpoint saved"; }
    [ $hot -ge 3 ] && fail "CPU at ${temp}C for 3 samples (warning level ${WARN_C}C); run still going"
    [ $slow -ge 5 ] && fail "CPU clock at ${perf}% of base for 5 samples (throttling); run still going"
  fi
  [ "$mem" -lt "$CRIT_MEM" ] && { stop_run; fail "only ${mem} MB memory free: stopped the run cleanly, checkpoint saved"; }
  [ "$mem" -lt "$WARN_MEM" ] && fail "only ${mem} MB memory free; run still going"
  [ "$disk" -lt "$MIN_DISK" ] && fail "only ${disk} GB disk free; run still going"
  [ "$idle" -ge "$STALL" ] && fail "no log activity for ${idle} min; the run looks stalled"
  n=$(bad_endings | wc -l)
  [ "$n" -gt "$base_bad" ] && fail "new bad game endings: $(bad_endings | tail -n $((n - base_bad)) | sort | uniq -c | tr -s ' \n' ' ')"
  n=$(warnings | wc -l)
  [ "$n" -gt "$base_warn" ] && fail "new fastchess warnings: $(warnings | tail -n $((n - base_warn)) | head -3 | tr '\n' '|')"
  sleep "$INTERVAL"
done
