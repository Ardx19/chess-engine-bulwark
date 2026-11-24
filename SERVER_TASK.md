# Task for the agent on the GPU server: GPU training only

You are running the neural-net training for Bulwark, a C++ chess engine (this repository), on a **shared** server.
**Scope: GPU training only.** Do not generate data, download Lc0 games, or run Stockfish on this server. The training
data (`data/training/` with `train/`, `val/`, `test/`) is built elsewhere and copied into this folder by the user.
Background if you need it: `README.md`, `docs/TRAINING.md`.

**Everything happens inside this project folder.** Run every command below from the project root. Checkpoints go
to `ckpt/`, nets to `nets/`, logs to `results/`. Nothing is written anywhere else.

## Ground rules (shared server, other people's jobs come first)

- Low priority always: prefix commands with `nice -n 19 ionice -c 3` (the queue script already does this).
- The only CPU use is what training needs: the build, the tests, and the trainer's data loader (capped at 4 threads).
- GPU: one GPU, one job at a time, through the job scheduler if there is one (single-GPU, lowest priority).
  `scripts/gpu_queue.sh` caps itself at 8% of GPU memory and 4 CPU threads, picks the emptiest GPU, and waits for
  free GPU memory before each run.
- Never kill, pause or touch other users' processes. Do not install system packages; use a Python venv.
- Do not push, upload or publish anything. Keep all nets on this server.
- Before starting the queue, report which GPU it will use and the expected time.

## 1. Set up and check (already done once; repeat only if something changed)

```bash
nvidia-smi; df -h .
python3 -m venv .venv && . .venv/bin/activate # or reuse an existing venv
pip install torch numpy chess pyyaml          # check: python3 -c "import torch; print(torch.cuda.is_available())"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && nice -n 19 cmake --build build -j4
./build/run_tests tests/data | tail -1                                 # expect "0 failures"
```

## 2. Wait for the data

Do nothing until `data/training/train/`, `data/training/val/` and `data/training/test/` exist and contain `.bin`
shards (and optionally `data/pilot_training/` for the sanity net). Then report the number of shards and total size.

## 3. Training on the GPU (~7-8 h if the GPU is free; low priority, resumable)

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

## 4. Report back when done (or when anything fails)

- `results/gpu_queue/queue.log`, the `verifynet` lines from `results/gpu_queue/<run>.log`, and
  `results/gpu_queue/evaluate.json`.
- Per run: final `val_logloss` from `ckpt/run-<name>/metrics.jsonl`, steps, wall time.
- Paths of the final nets: `nets/<run>/net-*.nnue` (+ `.nnuef`, `ref-*.jsonl`). Do not copy them anywhere else.
