#!/usr/bin/env bash
# Training notebook cell: !bash kaggle/train.sh   (committed "Save & Run All" runs keep /kaggle/working/ckpt)
# Env: RUN_ID, N (L1 width), HEAD (1 scalar / 3 WDL), STEPS, BATCH, DEADLINE (default 11h30m), DATA (shard dir), HF_REPO (optional)
set -euo pipefail
cd "$(dirname "$0")/.."
pip -q install huggingface_hub >/dev/null 2>&1 || true
python3 -m nnue.train --run-id "${RUN_ID:-n1wdl}" --ckpt-dir /kaggle/working/ckpt --n "${N:-256}" --head "${HEAD:-3}" \
  --steps "${STEPS:-200000}" --batch-size "${BATCH:-16384}" --lr "${LR:-0.001}" --deadline "${DEADLINE:-11h30m}" \
  --train-glob "${DATA:-/kaggle/input/nnue-data}/train/*.bin" --val-glob "${DATA:-/kaggle/input/nnue-data}/val/*.bin" \
  --resume auto
python3 -m nnue.export "/kaggle/working/ckpt/run-${RUN_ID:-n1wdl}/final.pt" /kaggle/working/nets \
  --ref-shard "$(ls ${DATA:-/kaggle/input/nnue-data}/val/*.bin | head -1)"
