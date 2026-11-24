#!/usr/bin/env bash
# All GPU training runs, one after another on ONE GPU, at low priority so other jobs on a shared server win.
#
#   DATA=~/data/nnue/training scripts/gpu_queue.sh            # run (or resume) the whole queue
#   RUNS="n1 n1wdl" scripts/gpu_queue.sh                      # only some runs
#   SMOKE=1 DATA=... scripts/gpu_queue.sh                     # 20 steps per run, to test the setup end to end
#
# Order: sanity (only if PILOT is set), n1, n1wdl, n2, n3count, n3learned (both N3 start from n2's final.pt).
# Each run resumes from its newest checkpoint if interrupted and is skipped once its DONE marker exists, so the queue
# can be killed at any time (kill -TERM the trainer: it checkpoints first) and restarted with the same command.
#
# Low priority:
#   * CPU: nice 19 and idle I/O class; TORCH_THREADS (default 4) caps PyTorch's CPU threads.
#   * GPU: CUDA has no process priority, so the queue stays polite instead: one GPU, one run at a time, at most
#     GPU_MEM_FRACTION of its memory (default 0.08, about 3.8 GB of an L40S; the nets need under 2 GB).
#     GPU picks the device; unset, it takes the GPU with the most free memory. Under a job scheduler, submit this
#     script as one low-priority single-GPU job instead (e.g. Slurm: sbatch --gres=gpu:1 --nice=10000 ...) and leave
#     GPU unset: the scheduler's CUDA_VISIBLE_DEVICES is respected.
#   * WAIT_FREE_GB (default 4): before each run, wait until the chosen GPU has this much free memory.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:?set DATA to the training root holding train/ val/ test/}
CKPT=${CKPT:-ckpt}
NETS=${NETS:-nets}
LOGS=${LOGS:-results/gpu_queue}
RUNS=${RUNS:-sanity n1 n1wdl n2 n3count n3learned}
BATCH=${BATCH:-16384}
export GPU_MEM_FRACTION=${GPU_MEM_FRACTION:-0.08}
export TORCH_THREADS=${TORCH_THREADS:-4}
export OMP_NUM_THREADS=$TORCH_THREADS MKL_NUM_THREADS=$TORCH_THREADS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
WAIT_FREE_GB=${WAIT_FREE_GB:-4}
mkdir -p "$CKPT" "$NETS" "$LOGS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/queue.log"; }

have_gpu() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }
if have_gpu && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU=${GPU:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)}
  export CUDA_VISIBLE_DEVICES=$GPU
fi
log "queue start: runs [$RUNS], GPU ${CUDA_VISIBLE_DEVICES:-none (CPU)}, mem fraction $GPU_MEM_FRACTION, threads $TORCH_THREADS"

wait_for_gpu() {
  have_gpu || return 0
  local dev=${CUDA_VISIBLE_DEVICES%%,*} free
  while :; do
    free=$(nvidia-smi -i "$dev" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 999999)
    [ "$free" -ge $((WAIT_FREE_GB * 1024)) ] && return 0
    log "GPU $dev has ${free} MiB free (< ${WAIT_FREE_GB} GiB); waiting"
    sleep 300
  done
}

# steps for the full run; SMOKE=1 shrinks everything to a quick end-to-end test
S() { if [ -n "${SMOKE:-}" ]; then echo 20; else echo "$1"; fi; }

args_for() {   # run name -> trainer arguments (positions seen = steps x BATCH)
  case $1 in
    sanity)    echo "--arch n1 --n 256 --head 3 --alpha 0.1 --steps $(S 2000)" ;;          # on the pilot data
    n1)        echo "--arch n1 --n 256 --head 1 --steps $(S 60000)" ;;                    # ~1.0B positions
    n1wdl)     echo "--arch n1 --n 256 --head 3 --alpha 0.1 --steps $(S 60000)" ;;        # ~1.0B
    n2)        echo "--arch n2 --n 512 --head 3 --alpha 0.1 --steps $(S 110000)" ;;       # ~1.8B
    n3count)   echo "--arch n3 --n 512 --head 3 --alpha 0.1 --k 8 --router count --steps $(S 45000) --init-from $CKPT/run-n2/final.pt" ;;      # ~0.74B
    n3learned) echo "--arch n3 --n 512 --head 3 --alpha 0.1 --k 8 --router learned --balance 0.01 --steps $(S 45000) --init-from $CKPT/run-n2/final.pt" ;;
    *) echo "unknown run $1" >&2; return 1 ;;
  esac
}

LOW=(nice -n 19)
command -v ionice >/dev/null 2>&1 && LOW+=(ionice -c 3)

for run in $RUNS; do
  if [ "$run" = sanity ] && [ -z "${PILOT:-}" ]; then log "skip sanity (set PILOT=<pilot training root> to run it)"; continue; fi
  if [ -e "$CKPT/run-$run/DONE" ]; then log "skip $run (done)"; continue; fi
  if [[ $run == n3* ]] && [ ! -e "$CKPT/run-n2/final.pt" ]; then log "skip $run: needs $CKPT/run-n2/final.pt"; continue; fi
  root=$DATA; [ "$run" = sanity ] && root=$PILOT
  wait_for_gpu
  log "start $run"
  # shellcheck disable=SC2046
  "${LOW[@]}" python3 -m nnue.train --run-id "$run" --ckpt-dir "$CKPT" --batch-size "$BATCH" --lr 0.001 --warmup 1000 \
      --val-every 2000 --train-glob "$root/train/*.bin" --val-glob "$root/val/*.bin" --resume auto \
      $(args_for "$run") > "$LOGS/$run.log" 2>&1 || { log "FAILED $run (see $LOGS/$run.log)"; exit 1; }
  if grep -q '"stopped_by_signal": true' "$LOGS/$run.log"; then log "stopped $run by signal; rerun to resume"; exit 0; fi
  ref=$(ls "$root"/val/*.bin | head -1)
  "${LOW[@]}" python3 -m nnue.export "$CKPT/run-$run/final.pt" "$NETS/$run" --ref-shard "$ref" --ref-n 10000 >> "$LOGS/$run.log" 2>&1
  if [ -x build/bulwark ]; then
    for f in "$NETS/$run"/net-*.nnuef "$NETS/$run"/net-*.nnue; do
      echo "verifynet $f $(ls "$NETS/$run"/ref-*.jsonl | head -1)" | "${LOW[@]}" build/bulwark | grep verifynet | tee -a "$LOGS/$run.log"
    done
  fi
  touch "$CKPT/run-$run/DONE"
  log "done $run -> $NETS/$run"
done

if [ -d "$DATA/test" ]; then
  finals=$(ls "$CKPT"/run-{n1,n1wdl,n2,n3count,n3learned}/final.pt 2>/dev/null || true)
  if [ -n "$finals" ]; then
    log "teacher agreement on the test split"
    # shellcheck disable=SC2086
    "${LOW[@]}" python3 -m nnue.evaluate "$DATA/test/*.bin" $finals --batch "$BATCH" ${SMOKE:+--max 100000} > "$LOGS/evaluate.json" 2>>"$LOGS/queue.log" \
      && log "wrote $LOGS/evaluate.json"
  fi
fi
log "queue finished"
