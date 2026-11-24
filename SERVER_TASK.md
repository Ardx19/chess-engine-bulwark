# Task for the agent on the GPU server

You are setting up and running the data generation and neural-net training for Bulwark, a C++ chess engine
(this repository). Everything is already coded and tested; your job is to run it, carefully, on a **shared** server.
Background if you need it: `README.md`, `docs/TRAINING.md`, `docs/pipeline.md`.

## Ground rules (shared server, other people's jobs come first)

- Low priority always: prefix every heavy command with `nice -n 19 ionice -c 3`.
- CPU: check the load first (`nproc`, `uptime`, `top -bn1 | head -20`). Use at most **half of the idle cores, and
  never more than 16**, for Stockfish workers (`WORKERS=...`). Lower it if the load from others goes up.
- GPU: one GPU, one job at a time, through the job scheduler if there is one (single-GPU, lowest priority / nice).
  `scripts/gpu_queue.sh` already caps itself at 8% of GPU memory, 4 CPU threads, and waits for free GPU memory.
- Never kill, pause or touch other users' processes. Do not install system packages; use a Python venv.
- Do not push, upload or publish anything. Keep all data and nets on this server.
- Before each long stage, report what you are about to run and how long you expect it to take.

## 0. Get the code

```bash
git clone bulwark.bundle bulwark && cd bulwark
```

## 1. Check the machine and set up (report the results)

```bash
nproc; uptime; free -g; df -h ~; nvidia-smi
which sbatch squeue qsub 2>/dev/null          # which scheduler, if any
python3 --version; cmake --version; g++ --version
grep -o -m1 avx2 /proc/cpuinfo                # empty -> add -DENGINE_ARCH=x86-64-v2 to the cmake line below
python3 -m venv ~/bulwark-venv && . ~/bulwark-venv/bin/activate
pip install torch numpy chess pyyaml          # CUDA build of torch; check: python3 -c "import torch; print(torch.cuda.is_available())"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && nice -n 19 cmake --build build -j8
./build/run_tests tests/data | tail -1                          # expect "0 failures"
python3 -m unittest discover -s tests -p 'test_*.py' 2>&1 | tail -3   # expect OK
```
Needs about 15 GB of free disk in `~` (data ~8 GB, tablebases ~1 GB, downloads are streamed and deleted).

## 2. Pilot (small, ~30-60 min): measures speed, downloads Stockfish 19 and the 3-5 piece tablebases

```bash
OUT=~/nnue/pilot SESSION=0 NUM_SESSIONS=1 MAX_TARS=3 RELABEL_LIMIT=20000 WORKERS=<cap> \
  nice -n 19 ionice -c 3 bash kaggle/pilot.sh 2>&1 | tee ~/nnue/pilot.log
python3 -m pipeline.tb_check "$HOME/nnue/pilot/master/*.bin" --oracle local --syzygy tools/syzygy   # must say "ok": true
python3 -m pipeline.check_splits --master "$HOME/nnue/pilot/labeled/*.bin"                          # must say "ok": true
# sanity-net data: all pilot positions (Lc0 + result labels; only 20k per shard were Stockfish-labelled)
python3 -m pipeline.make_training "$HOME/nnue/pilot/master/*.bin" ~/nnue/pilot_training --shard-records 200000
```
Report from `~/nnue/pilot/pilot_report-s000.json`: `acceptance`, `convert_positions_per_s`, the relabel
`seconds` / `records` (positions per second with your WORKERS), and `label_sanity.corr_expected_score` (should be
clearly positive, roughly 0.6 or more). Stop and report if any check fails.

## 3. Full dataset (~50M positions; CPU only, the long part: ~156 core-hours at 2,000 Stockfish nodes)

Done in 10 chunks of 13 tars, one chunk at a time, so a crash only loses the current chunk. Re-running a chunk
redoes that chunk only. Estimate the time per chunk from the pilot's relabel speed (about 5M positions per chunk).

```bash
for S in 0 1 2 3 4 5 6 7 8 9; do
  [ -e ~/nnue/full/manifest-s$(printf %03d $S).json ] && continue          # chunk already finished
  OUT=~/nnue/full SESSION=$S NUM_SESSIONS=10 MAX_TARS=13 RELABEL_LIMIT=0 WORKERS=<cap> STOP_H=1000 \
    nice -n 19 ionice -c 3 bash kaggle/pilot.sh 2>&1 | tee ~/nnue/full-s$S.log
done
python3 -m pipeline.tb_check "$HOME/nnue/full/master/*.bin" --oracle local --syzygy tools/syzygy --sample 2000
python3 -m pipeline.check_splits --master "$HOME/nnue/full/labeled/*.bin"
nice -n 19 python3 -m pipeline.make_training "$HOME/nnue/full/labeled/*.bin" ~/nnue/training
python3 -m pipeline.check_splits --training ~/nnue/training --max-overlap 0.05 --drop-overlap
```
`~/nnue/training` must contain `train/`, `val/`, `test/`. After `make_training` succeeds, `~/nnue/full/master` and
`~/nnue/full/work` can be deleted to save disk (keep `labeled/` and the manifests).

## 4. Training on the GPU (~7-8 h if the GPU is free; low priority, resumable)

First a 1-minute end-to-end check, then the real queue (sanity net, N1, N1-WDL, N2, N3-count, N3-learned):
```bash
SMOKE=1 DATA=~/nnue/training CKPT=~/nnue/smoke_ckpt NETS=~/nnue/smoke_nets LOGS=~/nnue/smoke_logs scripts/gpu_queue.sh
DATA=~/nnue/training PILOT=~/nnue/pilot_training CKPT=~/nnue/ckpt NETS=~/nnue/nets LOGS=~/nnue/logs \
  nohup scripts/gpu_queue.sh > ~/nnue/queue.out 2>&1 &
```
With a scheduler, submit the second command as one single-GPU lowest-priority job instead of `nohup` (Slurm example:
`sbatch --gres=gpu:1 --nice=10000 --cpus-per-task=4 --wrap "<the command>"`), and do not set `GPU`.
To stop: `kill -TERM` the `python3 -m nnue.train` process (it saves a checkpoint), then rerun the same command
to resume. Progress: `~/nnue/logs/queue.log` and `~/nnue/ckpt/run-<name>/metrics.jsonl`.

## 5. Report back when done (or when anything fails)

- `~/nnue/logs/queue.log`, the `verifynet` lines from `~/nnue/logs/<run>.log`, and `~/nnue/logs/evaluate.json`.
- Per run: final `val_logloss` from `metrics.jsonl`, steps, wall time.
- `~/nnue/full/manifest-s*.json` totals (positions kept, Stockfish version and sha256) and the pilot report numbers.
- Paths of the final nets: `~/nnue/nets/<run>/net-*.nnue` (+ `.nnuef`, `ref-*.jsonl`). Do not copy them anywhere else.
