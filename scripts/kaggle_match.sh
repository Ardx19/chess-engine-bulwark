#!/usr/bin/env bash
# Push N Kaggle CPU sessions (private script kernels) and fetch their results:
#   KIND=match (default): kaggle/match_runner.py, kernels ardx19/bulwark-match-s<i>, results -> results/kaggle_match/s<i>/
#   KIND=spsa:            kaggle/spsa_runner.py,  kernels ardx19/bulwark-spsa-s<i>,  results -> results/kaggle_spsa/s<i>/
#
#   scripts/kaggle_match.sh push  [SESSIONS=5 HOURS=3 CONC=3 TC=8+0.08 NEW=net-x.nnue BASE=net-y.nnue]
#   KIND=spsa scripts/kaggle_match.sh push  [SESSIONS=5 HOURS=3 CONC=4 NET=net-x.nnue NODES=25000 TOTAL_PAIRS=1500]
#   scripts/kaggle_match.sh status | fetch
# RESUME=1 attaches each kernel's own previous output, so a SPSA chain continues where the last version stopped.
# Needs the Kaggle token in .home/.kaggle/access_token and the ardx19/bulwark-match dataset (code bundle, nets, book,
# fastchess). Merge with: python3 scripts/merge_penta.py results/kaggle_match/s*/results.json [laptop logs]
set -euo pipefail
cd "$(dirname "$0")/.."
export KAGGLE_API_TOKEN=$(cat .home/.kaggle/access_token) KAGGLE_USERNAME=ardx19 KAGGLE_KEY=unused HOME=$PWD/.home
KAGGLE=${KAGGLE:-/home/aryandutt/miniconda3/bin/kaggle}
SESSIONS=${SESSIONS:-5}; HOURS=${HOURS:-3}; CONC=${CONC:-3}; TC=${TC:-8+0.08}
NEW=${NEW:-net-fcab4618.nnue}; BASE=${BASE:-net-a573c497.nnue}
KIND=${KIND:-match}; NET=${NET:-net-a573c497.nnue}; NODES=${NODES:-25000}; TOTAL_PAIRS=${TOTAL_PAIRS:-1500}
OUT=results/kaggle_$KIND

case ${1:-} in
  push)
    for i in $(seq 0 $((SESSIONS - 1))); do
      d=.tmp/kaggle_$KIND/s$i; rm -rf "$d"; mkdir -p "$d"
      sed -e "s/^SESSION = .*/SESSION = $i/" -e "s/^HOURS = .*/HOURS = $HOURS/" -e "s/^CONC = .*/CONC = $CONC/" \
          -e "s/^TC = .*/TC = \"$TC\"/" -e "s/^NEW = .*/NEW = \"$NEW\"/" -e "s/^BASE = .*/BASE = \"$BASE\"/" \
          -e "s/^NET = .*/NET = \"$NET\"/" -e "s/^NODES = .*/NODES = $NODES/" -e "s/^TOTAL_PAIRS = .*/TOTAL_PAIRS = $TOTAL_PAIRS/" \
          kaggle/${KIND}_runner.py > "$d/${KIND}_runner.py"
      ks="[]"; [ -n "${RESUME:-}" ] && ks="[\"ardx19/bulwark-$KIND-s$i\"]"
      cat > "$d/kernel-metadata.json" <<EOF
{"id": "ardx19/bulwark-$KIND-s$i", "title": "bulwark-$KIND-s$i", "code_file": "${KIND}_runner.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": false, "enable_tpu": false,
 "enable_internet": true, "dataset_sources": ["ardx19/bulwark-match"], "kernel_sources": $ks, "competition_sources": []}
EOF
      "$KAGGLE" kernels push -p "$d" 2>&1 | tail -1
    done ;;
  status)
    for i in $(seq 0 $((SESSIONS - 1))); do "$KAGGLE" kernels status "ardx19/bulwark-$KIND-s$i" 2>&1 | tail -1; done ;;
  fetch)
    for i in $(seq 0 $((SESSIONS - 1))); do
      mkdir -p "$OUT/s$i"
      "$KAGGLE" kernels output "ardx19/bulwark-$KIND-s$i" -p "$OUT/s$i" >/dev/null 2>&1 || echo "s$i: no output yet"
    done
    ls "$OUT"/s*/*.json 2>/dev/null ;;
  *) echo "usage: $0 push|status|fetch" >&2; exit 1 ;;
esac
