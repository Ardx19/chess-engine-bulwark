# Bulwark: project status (2026-11-24)

Bulwark is a bitboard C++ chess engine (UCI, Chess960, PVS search) being built into a defensive specialist. The plan is a self-trained W/D/L NNUE distilled from Lc0 and Stockfish, plus a draw-preferring utility. The claim to test: on held-out unbalanced positions the engine saves more games, at a measured strength cost.

Source of truth for scope: `PLAN_REMAINING.md` (uncommitted) and `project_B_nnue_chess_engine.md`. Frozen formats: `docs/interfaces.md`, `docs/pipeline.md`.

## Summary

| Phase | Status |
|---|---|
| 0-1 Board, move generation, Chess960, perft | Done |
| 2 UCI, search, transposition table, PeSTO eval, bench, tests | Done (UCI protocol tests added) |
| 3 Selectivity ladder and defense switches | Done: IIR on, README, scripts, results registry |
| 4 Data and training pipeline | Code done incl. data checks; **not yet run on real data** |
| 5 NNUE in the engine | Code done (incremental, AVX2, EvalFile); **no trained net yet** |
| 6 Defense study (D1/D2, benchmark sets) | Code done (D1/D2/D6/D7, harness, stats, suites); **needs the net and the runs** |
| 7-8 N2, MoE | Code done (trainer, format v2, C++ inference, parity tests); **needs training** |
| 9 Write-up | Waiting on results |

**Update 2026-11-24.** All coding parts of the plan are implemented and tested; what remains is data generation,
training (see `docs/TRAINING.md`) and the measurement runs. Bench is now **137,089** (IIR on).

## Engine: what works

**Build and tests**
- CMake build with `bulwark`, `perft`, `run_tests`, `lc0_convert` targets. Release flags `-O3`, `-march=x86-64-v3`.
- C++ suite (`build/run_tests tests/data`): 113,820 checks, 0 failures. Includes the Syzygy tablebase test when `SYZYGY_PATH` is set.
- Perft exact on the standard CPW positions and on 241 Chess960 positions checked against python-chess. ASan/UBSan debug build exists.
- Movegen speed about 86 Mnps (from Phase 1 notes).
- `bulwark bench` is deterministic: **137,089 nodes** with IIR on (163,284 at 8cd930c). Speed about 0.97 Mnps in bench.

**UCI and search**
- UCI commands: `position` (startpos/fen, moves), `go` (depth, nodes, movetime, clock, infinite), `stop`, `ucinewgame`, `setoption`, `isready`, `quit`.
- Options: `Hash`, `Move Overhead`, `UCI_Chess960`, `SyzygyPath`, and one switch per feature (below).
- Search: PVS with transposition table, aspiration windows, reverse futility pruning, null-move pruning, late move reductions, killers and history, LMP and futility, SEE pruning, improving flag, IIR (switch, see below), time management with an optional scaling switch.
- Syzygy probing through the vendored Pyrrhic library. Checked against python-chess on 400 positions.

**Feature switches (all UCI options, default states)**

| Switch | Default | Measured |
|---|---|---|
| Aspiration | on | +21 Elo, no verdict (600 games) |
| RFP | on | +53, no verdict |
| NMP | on | +80, **H1 accepted** |
| LMR | on | +148, **H1 accepted** |
| KillersHistory | on | +149, **H1 accepted** |
| LMPFutility | on | +48, no verdict |
| IIR | **off** (see below) | −15 in ladder; re-test +21.7 ± 14.1 over 1,200 games |
| Improving | on | +8, no verdict |
| SEEPrune | on | +56, **H1 accepted** |
| TMScaling | on | +19.7 ± 20.2 (600 games), no verdict |
| D3_Rule50Scale | on | +7.5 ± 20.7 (600 games), inconclusive |
| D4_Syzygy | on | +35.5 ± 19.1 (600 games), close to passing |
| D5_StalemateGuard | on | −9.3 ± 19.0 (600 games), inconclusive, slightly negative |

**Decisions from the overnight results** (recorded in `results/overnight-20261020-0332/REPORT.txt`):
- **IIR is now on** (commit d160dd8). The re-test over 1,200 games gave +21.7 ± 14.1, reversing the earlier −15.
- **D5 is slightly negative** but inconclusive. It stays on until more evidence.

**Strength benchmark**
- Final build vs Phase 2 build: **+407 ± 49 Elo** (400 games, 91% score). This is the total gain from the selectivity work.
- Against Stockfish 19 with `UCI_LimitStrength`: 600 games at 60+0.6 across levels 2600/2800/3000.
  - vs 2600: 70.2% score (+149 Elo)
  - vs 2800: 44.2% (−40 Elo)
  - vs 3000: 18.0% (−263 Elo)
  - **Estimated rating about 2750 ± 31** (CCRL 40/4 scale). The earlier 2000–2600 levels were too low to be useful and were discarded in the report.
- The rating was computed with a maximum-likelihood fit, not Ordo (which the plan specified and isn't installed).

**Stability and legality**
- 1,000-game self-play at 8+0.08 (tablebases on): **no illegal moves, no time losses.**
- 400 Chess960 games from 1,000 generated DFRC starts: **no illegal moves, no time losses.**
- 600 benchmark games: no illegal moves, no time losses.
- Phase 2 gate asked for 1,000 games; that's now met.

## Tooling

- **fastchess** 1.8.2 at `~/tools/fastchess/fastchess`. Stockfish 19 at `~/tools/stockfish/stockfish/stockfish-linux-x86-64-universal` (sha256 in `~/tools/stockfish/SF19.sha256`).
- **Opening book:** `~/data/books/8moves_v3.pgn`. **DFRC openings:** `~/data/books/dfrc_1000.epd` (generated by `scripts/gen_dfrc.py`).
- **Syzygy:** 3-5 piece tables at `~/data/syzygy/tb345` (about 940 MB).
- **Scripts** (`scripts/`):
  - `feature_sprt.sh`: one-switch SPRT of one feature against the default build.
  - `ladder.sh`, `sprt.sh`, `soak.sh`: ladder steps, timed SPRT, legality soak.
  - `overnight.sh`: unattended queue with per-stage checkpoints. Resumable with `RUN=<dir> scripts/overnight.sh`.
  - `watchdog.sh`: monitors CPU temperature and clock (through Windows), memory, C: drive space, stalls, bad game endings and fastchess warnings. Stops the run cleanly on critical heat or memory.
  - `overnight_report.py`: builds `REPORT.txt` from a run directory.
  - `gen_perft960.py`, `divide_diff.py`: perft reference and debugging.

## Phase 4: data and training pipeline

**Code written, C++ and Python tests passing. Not yet run on real data.**

- **Converter** (`tools/lc0_convert.cpp`, `pipeline/convert.py`): reads Lc0 v6 `test91` training tars into 56-byte master records. Filters: fewer than 8 plies of history, in check, capture or promotion best move, |root_q| > 0.98, malformed or illegal positions, deletion mark. Measured about 20k positions/s, 64% acceptance.
- **Stockfish relabel** (`pipeline/relabel.py`): scores each position with Stockfish 19 for W/D/L. About 89 relabels/s per worker under load.
- **Tablebase override**: positions with 5 or fewer pieces, rule50 = 0, no castling get the exact outcome.
- **Manifest** (`pipeline/manifest.py`): records source, Stockfish version and hash, filter counts, and shard hashes.
- **Splits and dedup** (`pipeline/make_training.py`, `pipeline/records.py`): split by game id, 40-byte training records.
- **Loader and trainer** (`nnue/data.py`, `nnue/train.py`, `nnue/model.py`, `nnue/targets.py`): resumable cursor, checkpoint and exact-resume tested. N1 (scalar) and N1-WDL (3 logits) heads.
- **Export** (`nnue/export.py`): `.nnue` (format `NNUEWDL1`, arch IDs N1=1, N1-WDL=2, N2=3, N3=4, QA=255, QB=64) plus `.nnuef` and reference JSONL.
- **C++ inference** (`src/nnue.cpp`, `src/nnue.h`): float and quantized paths. `verifynet` command.
  - Float matches PyTorch to 1.4e-6.
  - Quantized within 0.055 logits (about 5 cp mean error).
- **Tests:** 17 Python pipeline tests pass (one skipped without the real Lc0 tar, `LC0_TAR`). C++ suite passes.

**Not done:** Kaggle pilot and production data runs, GPU training, the tablebase-override check against the Lichess API, the leakage check, bullet cross-check converter, any trained net.

## Added on 2026-11-24 (all committed)

- Phase 3: IIR on; `tests/test_uci.py` (13 protocol tests incl. Chess960 castling spelling); `go searchmoves`;
  `scripts/netcheck.sh`, `scripts/strength.sh`, `scripts/registry.py` + `results/registry.csv` (seeded with all past runs);
  `README.md`.
- Phase 4: `pipeline/tb_check.py` (labels vs Lichess API / local tables), `pipeline/check_splits.py` (session reuse,
  val/test overlap with train), `pipeline/to_bullet.py`; fixed the pilot's Syzygy download (tables went to /dev/null).
- Phase 5: `EvalFile`; per-ply accumulator stack updated from board diffs; AVX2 output kernel; exact parity with the
  scalar reference; `eval` prints net W/D/L; `bench <depth> <net>`.
- Phase 6: D1 (`D1_WDLUtility`, `DrawValue`), D2 (`DefenseMode`, `DefenseDrawValue`, `DefenseLossThreshold`),
  D6 (`D6_PerpetualSeek`), D7 (`D7_FortressDetect`), `UCI_ShowWDL`; TT cleared on utility changes; benchmark-set carving,
  save-rate harness, paired statistics, calibration metrics, tablebase and fortress suite tools, teacher-agreement
  evaluator (`nnue/evaluate.py`).
- Phases 7-8: N2 and N3 (piece-count and learned routers, upcycling, load-balancing loss) in the trainer; export
  format v2; C++ inference with king buckets, Finny tables, int8 L2/router kernels; float parity with PyTorch < 1e-4.
- Measured training speed on this laptop's CPU: N1 ~45k positions/s, N2/N3 ~20k/s (`docs/TRAINING.md`).
- Search speed with a net (random weights, bench depth 6): N1 256 ~0.73 Mnps, N2 512 ~0.31 Mnps (HCE ~1.0 Mnps).

## Repository

- Branch `master`; everything is committed. Overnight outputs in `results/` are gitignored except `results/registry.csv`.

## Infrastructure and environment

- **Laptop:** Windows with WSL2 Ubuntu, 12 logical CPUs, about 12 GB RAM visible to WSL. C: drive about 19 GB free of 476 GB, which is the real limit for WSL's disk.
- **Power:** Windows Modern Standby froze WSL for about 4 hours (08:28–12:38) despite "sleep: Never". The overnight script now asks Windows to stay awake while it runs (`SetThreadExecutionState`, released when the run ends).
- **Thermals:** under full 5-game concurrency the CPU runs at 80–91 °C and about 130–165% of base clock.
- **Server:** 32 logical CPUs, 251 GB RAM, shared with other workloads. Not yet accessed. Planned use: Stockfish relabeling for the training data (see `memory/server-for-data-generation`).

## Known issues and debt

- **Ladder deviations:** 4+0.04 time control and a 600-game cap, so several steps have no SPRT verdict. Point estimates stay on.
- **Threads** is fixed at 1; no pondering.
- **Syzygy root filter** only applies at 5 pieces or fewer.
- **Quantization error:** about 5 cp mean; may cost Elo at Phase 5. QAT is the fallback.
- **Position size** about 700 bytes (copy-make); may matter for NPS once accumulators land.
- **Lc0 data** mixes in Chess960 games; castling rights are dropped when not on standard squares; en passant is not in Lc0 format 1.
- **Benchmark rating** used an internal maximum-likelihood fit instead of Ordo.
- **Stockfish stage** was interrupted by Windows standby once (08:28–12:38); the checkpoint resumed cleanly.

## Next steps

1. Data: pilot run, data checks, then the full relabel (server or Kaggle CPU), `make_training`.
2. Train N1 and N1-WDL (`docs/TRAINING.md`), export, `verifynet`, register in `nets/registry.csv`.
3. NNUE vs HCE at fixed nodes and fixed time; retune pruning margins if the eval scale shifts; strength benchmark.
4. Benchmark sets from UHO, `N_sf` calibration, defense study on dev then test, fortress / tablebase suites.
5. Stretch: N2, then N3 variants; equal-active-compute table.
6. Open engineering debt: NPS with nets (explicit SIMD for accumulator updates), Threads = 1, Ordo for ratings,
   curating the fortress suite (no public suite exists), top-2 routing (optional N3 variant D).
