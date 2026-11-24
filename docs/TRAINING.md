# What has to be trained, and what it needs

Nothing has been trained yet: every net below starts from random weights, except N3, which is "upcycled" from the
trained N2. All code is in place and tested on synthetic data (`tests/test_pipeline.py`, `build/run_tests`). The
trainer is device-agnostic: it uses CUDA when PyTorch sees a GPU and the CPU otherwise (`--device cpu|cuda` forces one).

## 0. The dataset (needed before any net)

Not a training run, but nothing can be trained without it. One dataset feeds every net.

| Step | Command | Cost (measured where stated) |
|---|---|---|
| Download + convert Lc0 `test91` tars (input format 1, v6), with the tablebase label override | `python3 -m pipeline.run --run test91 --session S --num-sessions N --max-tars T --stockfish $SF --syzygy $TB` (or `kaggle/pilot.sh`) | ~130 MB per tar, ~635k records, 64% kept (~400k positions); convert ~20k positions/s (measured). 50M positions ≈ 125 tars ≈ 16 GB streamed (each tar is deleted after conversion) |
| Stockfish 19 relabel (W/D/L + cp), 2,000 nodes | same driver (`--sf-nodes`, `--relabel-limit 0`) | ~89 positions/s per core under load (measured). 50M positions ≈ 156 core-hours: ~40 h on 4 cores, ~6 h on 28 cores |
| Pilot first | `MAX_TARS=3 RELABEL_LIMIT=20000 bash kaggle/pilot.sh` (works on any Linux box) | Minutes. Sets the node budget and dataset size from real throughput |
| Data checks | `python3 -m pipeline.tb_check 'master/*.bin' --oracle lichess`; `python3 -m pipeline.check_splits --master 'master/*.bin'` | Minutes (the API check is throttled to ~1 request/s) |
| Split by game, dedup, 40-byte records | `python3 -m pipeline.make_training 'master/*.bin' data/training` then `python3 -m pipeline.check_splits --training data/training --max-overlap 0.05` | Minutes |

Target: **20-50M positions** for N1/N1-WDL/N2/N3. Disk: ~56 B/position master (2.8 GB at 50M) + 40 B training
(2 GB). Grow toward 300M only for N2/N3, and only if validation loss is still improving.

## 1. The nets, in order

| # | Net | Why | Architecture | Train command (add `--train-glob`, `--val-glob`, `--run-id`, `--ckpt-dir`) | Gate before the next |
|---|---|---|---|---|---|
| 0 | **Sanity net** | Proves the pilot data trains | N1-WDL, L1 64-256 | `python3 -m nnue.train --arch n1 --n 256 --head 3 --steps 2000` on pilot data | Loss falls; `verifynet` passes |
| 1 | **N1** (scalar) | First NNUE; Phase 5 NNUE-vs-HCE gain | 768 → 256 ×2, SCReLU → 1 | `--arch n1 --n 256 --head 1` | Beats HCE at fixed nodes and fixed time |
| 2 | **N1-WDL** | The fixed net for the whole defense study (D1/D2 need W/D/L) | same body → 3 (W/D/L) | `--arch n1 --n 256 --head 3 --alpha 0.1` | Not weaker than N1 at fixed nodes; calibration sane |
| 3 | (optional) bullet N1 | Cross-checks our trainer against bullet's `simple` example | same as N1 | `python3 -m pipeline.to_bullet 'train/*.bin' bullet.txt`, then bullet-utils + bullet | Similar strength to our N1 |
| 4 | **N2** (stretch) | Capacity: king buckets + hidden layers | 8 mirrored king buckets ×768 → 512 → 16 → 32 → 3 | `--arch n2 --n 512 --head 3` | SPRT vs N1-WDL at neutral utility |
| 5 | **N3-B** (stretch) | MoE, hard routing by piece count | N2 body, K = 8 stacks | `--arch n3 --n 512 --k 8 --router count --init-from <N2 final.pt>` | Equal-active-compute table |
| 6 | **N3-C** (stretch) | MoE, learned top-1 router + load balancing | N2 body, K = 8 stacks + router | `--arch n3 --n 512 --k 8 --router learned --balance 0.01 --init-from <N2 final.pt>` | Same table; a negative result is acceptable |

N3 variant A (dense) is N2 itself: one stack of the same size, so the active compute per evaluation is equal across
A/B/C. Top-2 routing (optional variant D) is not implemented.

After each run:
```bash
python3 -m nnue.export ckpt/run-<id>/final.pt nets --ref-shard data/training/val/s-00000.bin   # .nnue + .nnuef + ref-*.jsonl
echo "verifynet nets/net-XXXX.nnuef nets/ref-XXXX.jsonl" | build/bulwark                       # float: max error < 1e-4
echo "verifynet nets/net-XXXX.nnue  nets/ref-XXXX.jsonl" | build/bulwark                       # quantised: < ~0.15 logits
python3 -m nnue.evaluate 'data/training/test/*.bin' ckpt/run-*/final.pt                        # teacher agreement on test
build/bulwark bench 13 nets/net-XXXX.nnue                                                      # speed with the net
```
Then add a row to `nets/registry.csv` (lineage: run, data manifest, positions seen, metrics, parent net).

## 2. How long each run takes

Measured on this laptop's CPU (Ryzen 5 5500U, 6 cores / 12 threads, PyTorch CPU, batch 16,384):

| Net | Training throughput | Data-loader ceiling (one thread, measured) |
|---|---|---|
| N1 / N1-WDL (L1 256) | ~45k positions/s | ~390k positions/s |
| N2 / N3 (L1 512) | ~20k positions/s | ~190k positions/s |

On a GPU the step itself is much cheaper, so expect throughput close to the loader ceiling (not measured on a GPU
yet). How many positions to show each net is a choice; a reasonable start with 50M positions:

| Net | Positions seen (≈ epochs) | Laptop-class CPU | 16-32 core server CPU | Any CUDA GPU (loader-bound) |
|---|---|---|---|---|
| Sanity net | 30M | ~10 min | minutes | minutes |
| N1, N1-WDL (each) | ~1B (20 epochs) ≈ 60k steps | ~6 h | ~2-3 h (estimate) | ~1 h |
| N2 | ~1.5-2B ≈ 100k steps | ~1 day (too slow) | ~6-10 h (estimate) | ~2-3 h |
| N3-B, N3-C (each, upcycled) | ~0.5-1B | ~10 h | ~4-6 h (estimate) | ~1-1.5 h |

Before committing to a device, run the 25-step speed test there and scale from it:
```bash
python3 -m nnue.train --run-id speed --arch n1 --n 256 --head 3 --steps 25 --warmup 2 --val-every 1000 \
  --train-glob 'data/training/train/*.bin' --resume none        # positions/s = 25 * 16384 / wall seconds
```
`--deadline 11h30m` makes any run stop cleanly (checkpoint + final export) before a session limit; `--resume auto`
continues from the newest checkpoint (locally, `/kaggle/input/*/ckpt`, or `$HF_REPO`), with an identical loss curve.

## 3. What is required

| Item | Needed for | Notes |
|---|---|---|
| Python 3.10+, PyTorch ≥ 2.1, NumPy | all training | CUDA build of PyTorch only if a GPU is used |
| python-chess | tests, benchmark sets, save-rate harness, data checks | `pip install chess` |
| PyYAML, huggingface_hub | optional | `--config` files; checkpoint mirroring (`HF_REPO`, token in `HF_TOKEN`) |
| CMake + a C++20 compiler (g++ ≥ 10) | `lc0_convert`, the engine (`verifynet`, `bench`) | `-DENGINE_ARCH=x86-64-v3` needs AVX2; use `x86-64-v2` on older CPUs |
| Stockfish 19 binary (pinned, hash recorded) | relabeling, benchmark sets, save-rate opponent | the hash goes into the dataset manifest |
| Syzygy 3-5 piece tables (~1 GB) | tablebase label override, adjudication | `kaggle/pilot.sh` downloads them |
| Internet | Lc0 tars (storage.lczero.org), Lichess tablebase API checks | |
| CPU cores | relabeling dominates: ~156 core-hours for 50M positions | more cores = proportionally faster |
| Disk | ~5 GB per 50M positions (master + training) plus ~1 GB of tables | tars are streamed, not kept |
| RAM | 4 GB is enough for training (shards are read one at a time, ~200 MB each) | relabel workers are ~50 MB each |
| GPU | optional | any CUDA GPU with ≥ 2 GB; the nets are tiny, the data loader is the limit |

## 4. What is measured after training (not training, but it needs the nets)

1. NNUE vs HCE: fixed-node SPRT (`scripts/netcheck.sh`-style, or `scripts/sprt.sh` with `option.EvalFile`), then timed.
   Retune the RFP/futility/LMR margins if the eval scale shifts.
2. Benchmark sets: `scripts/make_benchsets.py` on `UHO_Lichess_4852_v1.epd` (download it first), commit `benchsets/`.
3. Calibrate `N_sf` on the calibration set with N1-WDL at neutral settings (target ~30% save rate).
4. Defense study on dev, then once on test: `scripts/saverate.py` + `scripts/saverate_stats.py`; D3-D5 ablations;
   non-regression SPRT `[-5, 0]` with D1/D2 on; calibration (`scripts/calibration.py`); fortress and tablebase suites.
5. Strength benchmark again with the final net (`scripts/strength.sh`).
