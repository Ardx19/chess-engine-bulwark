# Task for the agent on the GPU server

You are running the data generation and neural-net training for Bulwark, a C++ chess engine (this repository).
Everything is coded and tested; your job is to run it, carefully, on a **shared** server.
Background if you need it: `README.md`, `docs/TRAINING.md`, `docs/pipeline.md`.

**Everything happens inside this project folder.** Run every command below from the project root. Downloads go to
`tools/`, data to `data/`, checkpoints to `ckpt/`, nets to `nets/`, logs to `results/`. Nothing is written anywhere else.

## Ground rules (shared server, other people's jobs come first)

- Low priority always: prefix every heavy command with `nice -n 19 ionice -c 3`.
- CPU: check the load first (`nproc`, `uptime`, `top -bn1 | head -20`). Use at most **half of the idle cores, and
  never more than 16**, for Stockfish workers (`WORKERS=...`). Lower it if the load from others goes up.
- GPU: one GPU, one job at a time, through the job scheduler if there is one (single-GPU, lowest priority).
  `scripts/gpu_queue.sh` caps itself at 8% of GPU memory and 4 CPU threads, picks the emptiest GPU, and waits for
  free GPU memory before each run.
- Never kill, pause or touch other users' processes. Do not install system packages; use a Python venv.
- Do not push, upload or publish anything. Keep all data and nets on this server.
- Before each long stage, report what you are about to run, the cores/GPU it will use, and the expected time.

## 1. Set up and check (report the results)

```bash
nproc; uptime; free -g; df -h .; nvidia-smi
grep -o -m1 avx2 /proc/cpuinfo                # empty -> add -DENGINE_ARCH=x86-64-v2 to the cmake line below
python3 -m venv .venv && . .venv/bin/activate # or reuse an existing venv
pip install torch numpy chess pyyaml          # check: python3 -c "import torch; print(torch.cuda.is_available())"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && nice -n 19 cmake --build build -j8
./build/run_tests tests/data | tail -1                                 # expect "0 failures"
python3 -m unittest discover -s tests -p 'test_*.py' 2>&1 | tail -3    # expect OK
```
Needs about 20 GB of free disk here (data ~10 GB, tablebases ~1 GB; Lc0 tars are downloaded one at a time and
deleted after conversion).

## 2. Pilot (small): measures speed, downloads Stockfish 19 and the 3-5 piece tablebases

```bash
OUT=data/pilot SESSION=0 NUM_SESSIONS=1 MAX_TARS=3 RELABEL_LIMIT=20000 WORKERS=<cap> \
  nice -n 19 ionice -c 3 bash kaggle/pilot.sh 2>&1 | tee results/pilot.log
python3 -m pipeline.tb_check 'data/pilot/master/*.bin' --oracle local --syzygy tools/syzygy   # must say "ok": true
python3 -m pipeline.check_splits --master 'data/pilot/labeled/*.bin'                          # must say "ok": true
# sanity-net data: all pilot positions (Lc0 + result labels; only 20k per tar were Stockfish-labelled)
python3 -m pipeline.make_training 'data/pilot/master/*.bin' data/pilot_training --shard-records 200000
```
The 3 tars are picked at random across the Lc0 run (they range from ~50 MB to ~1 GB each). Report from
`data/pilot/pilot_report-s000.json`: `totals.kept`, `acceptance`, `convert_positions_per_s`, the relabel `seconds` and
`records` (Stockfish positions per second with your WORKERS), and `label_sanity.corr_expected_score` (clearly
positive, roughly 0.6 or more). Stop and report if any check fails.

## 3. Full dataset: ~50M positions in 10 chunks of 5M (CPU only, the long part: ~156 core-hours)

Each chunk takes its own tars (disjoint from the other chunks, spread across the run) until 5M positions are kept,
then Stockfish-labels them. A crash only loses the current chunk; rerunning skips finished chunks. Estimate the
time per chunk from the pilot's Stockfish speed: 5,000,000 / (positions per second).

```bash
for S in 0 1 2 3 4 5 6 7 8 9; do
  [ -e data/full/manifest-s$(printf %03d $S).json ] && continue          # chunk already finished
  OUT=data/full SESSION=$S NUM_SESSIONS=10 MAX_TARS=500 MAX_KEPT=5000000 RELABEL_LIMIT=0 WORKERS=<cap> STOP_H=1000 \
    nice -n 19 ionice -c 3 bash kaggle/pilot.sh 2>&1 | tee results/full-s$S.log
done
python3 -m pipeline.tb_check 'data/full/master/*.bin' --oracle local --syzygy tools/syzygy --sample 2000
python3 -m pipeline.check_splits --master 'data/full/labeled/*.bin'
nice -n 19 python3 -m pipeline.make_training 'data/full/labeled/*.bin' data/training
python3 -m pipeline.check_splits --training data/training --max-overlap 0.05 --drop-overlap
```
`data/training` must contain `train/`, `val/`, `test/`. After `make_training` succeeds, `data/full/master` and
`data/full/work` can be deleted to save disk (keep `data/full/labeled/` and the manifests).

## 4. Training on the GPU (~7-8 h if the GPU is free; low priority, resumable)

First a 1-minute end-to-end check, then the real queue (sanity net, N1, N1-WDL, N2, N3-count, N3-learned).
The queue reads `data/training` (and `data/pilot_training` for the sanity net) by default.
```bash
SMOKE=1 CKPT=ckpt_smoke NETS=nets_smoke LOGS=results/gpu_smoke scripts/gpu_queue.sh
nohup scripts/gpu_queue.sh > results/gpu_queue.out 2>&1 &
```
With a scheduler, submit the second command as one single-GPU lowest-priority job instead of `nohup` (Slurm example:
`sbatch --gres=gpu:1 --nice=10000 --cpus-per-task=4 --wrap "cd $PWD && scripts/gpu_queue.sh"`).
To stop: `kill -TERM` the `python3 -m nnue.train` process (it saves a checkpoint), then rerun the same command to
resume. Progress: `results/gpu_queue/queue.log` and `ckpt/run-<name>/metrics.jsonl`.

## 5. Report back when done (or when anything fails)

- `results/gpu_queue/queue.log`, the `verifynet` lines from `results/gpu_queue/<run>.log`, and
  `results/gpu_queue/evaluate.json`.
- Per run: final `val_logloss` from `ckpt/run-<name>/metrics.jsonl`, steps, wall time.
- `data/full/manifest-s*.json` totals (positions kept, Stockfish version and sha256) and the pilot report numbers.
- Paths of the final nets: `nets/<run>/net-*.nnue` (+ `.nnuef`, `ref-*.jsonl`). Do not copy them anywhere else.
