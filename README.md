# Bulwark

A bitboard C++ chess engine (UCI, standard chess and Chess960/DFRC, PVS alpha-beta) being built into a **defensive
specialist**: a self-trained W/D/L NNUE distilled from Lc0 and Stockfish, plus a draw-preferring utility that is
measured, not assumed. The claim under test: on held-out unbalanced positions, a draw-aware utility on a fixed net
saves more games, at a measured strength cost.

Scope and statistics: `project_B_nnue_chess_engine.md`. Remaining work: `PLAN_REMAINING.md`. Status: `docs/STATUS.md`.
Training guide: `docs/TRAINING.md`. Frozen formats: `docs/interfaces.md`, `docs/pipeline.md`.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/run_tests tests/data                     # C++ suite (perft, SEE, eval symmetry, search, NNUE parity); --full for deep perft
SYZYGY_PATH=~/data/syzygy/tb345 ./build/run_tests tests/data   # adds the tablebase checks
python3 -m unittest discover -s tests -p 'test_*.py'          # UCI protocol + data/training pipeline tests
./build/bulwark bench                            # deterministic node count, quoted as "Bench:" in every commit
```

Default target is `-march=x86-64-v3` (AVX2 + BMI2). Debug build with ASan/UBSan: `-DCMAKE_BUILD_TYPE=Debug`.

## Results so far (handcrafted eval)

All matches with fastchess, 8moves_v3 book, single thread, Hash 16. Full rows in `results/registry.csv`.

**Selectivity ladder** (4+0.04, SPRT [0, 10], game cap 600; each step against the previous step):

| Step | Elo | Games | Verdict |
|---|---|---|---|
| 1 Aspiration windows | +20.9 ± 20.6 | 600 | no verdict |
| 2 Reverse futility pruning | +52.5 ± 23.5 | 600 | no verdict |
| 3 Null-move pruning | +79.6 ± 29.4 | 444 | H1 |
| 4 Late-move reductions | +147.5 ± 37.8 | 252 | H1 |
| 5 Killers + history | +148.9 ± 35.8 | 250 | H1 |
| 6 LMP + futility | +47.8 ± 21.0 | 600 | no verdict |
| 7 IIR | −15.1 ± 21.8 (re-test below: +21.7 ± 14.1) | 600 | no verdict |
| 8 Improving flag | +8.1 ± 20.3 | 600 | no verdict |
| 9 SEE pruning | +55.7 ± 21.6 | 516 | H1 |

**One-switch tests** (4+0.04; new = switch on, base = off):

| Switch | Elo | Games | Bounds | Default |
|---|---|---|---|---|
| TMScaling (time management) | +19.7 ± 20.2 | 600 | [−5, 5] | on |
| D3_Rule50Scale | +7.5 ± 20.7 | 600 | [−5, 5] | on |
| D4_Syzygy | +35.5 ± 19.1 | 600 | [−5, 5] | on |
| D5_StalemateGuard | −9.3 ± 19.0 | 600 | [−5, 5] | on (inconclusive) |
| IIR (re-test) | +21.7 ± 14.1 | 1200 | [0, 10] | **on** |

**Whole ladder:** final build vs the Phase 2 build: **+407 ± 49 Elo** (400 games).

**Strength vs Stockfish 19 `UCI_LimitStrength`** (60+0.6, 600 games at `UCI_Elo` 2600 / 2800 / 3000): scores 70.2% /
44.2% / 18.0%, a maximum-likelihood estimate of **about 2750 ± 31** on the `UCI_Elo` scale (not a CCRL rating).

**Robustness:** 1,000 self-play games at 8+0.08 and 400 DFRC games: no illegal moves, no time losses.

**Speed** (Ryzen 5 5500U, WSL2, one thread): perft 37 Mnps (startpos depth 6), 52 Mnps (Kiwipete depth 5);
search about 1.0 Mnps with the handcrafted eval. `Bench: 137089`.

No net has been trained yet; the NNUE, defense-study and N2/N3 results are the next rows of this README.

## UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash`, `Move Overhead`, `UCI_Chess960`, `SyzygyPath` | 16, 10, false, empty | standard |
| `Threads` | 1 | fixed at 1 |
| `EvalFile` | empty | `.nnue` (quantised, for play) or `.nnuef` (float, parity only); empty = handcrafted eval |
| `Aspiration` `RFP` `NMP` `LMR` `KillersHistory` `LMPFutility` `IIR` `Improving` `SEEPrune` `TMScaling` | true | ladder switches |
| `D3_Rule50Scale`, `D4_Syzygy`, `D5_StalemateGuard` | true | defense switches measured above |
| `D1_WDLUtility` + `DrawValue` (per mille) | false, 500 | utility `U = W + d·D` with `d = DrawValue` (opponent nodes use `1 − d`); in-tree draws score `400·logit(d)` |
| `DefenseMode` (D2) + `DefenseDrawValue`, `DefenseLossThreshold` | false, 650, 400 | `d = DefenseDrawValue` when the last search's P(loss) ≥ threshold |
| `D6_PerpetualSeek` | false | quiet checks at the first qsearch ply when far below alpha (experimental) |
| `D7_FortressDetect` | false | flat large score with no zeroing move in the PV halves the distance to the draw score (experimental) |
| `UCI_ShowWDL` | false | net W/D/L at the end of the PV in `info` lines (WDL nets) |

Extra commands: `bench [depth] [net] [verbose]` (also from the shell), `eval`, `d`, `perft <depth>`,
`verifynet <net> <ref.jsonl> [limit]`, `go ... searchmoves ...`.

## Evaluation nets

| Net | Inputs | Body | Head | Format |
|---|---|---|---|---|
| N1 / N1-WDL | 768 per perspective | 256, SCReLU | 1 scalar / 3 W/D/L | v1 (arch 1 / 2) |
| N2 | 8 mirrored king buckets × 768 | 512, SCReLU → 16 → 32 | 3 | v2 (arch 3) |
| N3 | as N2 | K = 8 stacks, router: piece count or learned top-1 | 3 | v2 (arch 4) |

Accumulators are updated incrementally on a per-ply stack (Finny-table refresh on king bucket changes), with AVX2
kernels that equal the scalar reference exactly; float inference matches PyTorch to < 1e-4.

## Defense-study tooling

`scripts/make_benchsets.py` (frozen calibration/dev/test/sealed sets from UHO), `scripts/saverate.py` (engine plays the
worse side vs Stockfish), `scripts/saverate_stats.py` (paired cluster-bootstrap CIs, exact discordant-pair test,
transition table), `scripts/calibration.py` (log loss, Brier, ECE, reliability diagram), `scripts/tb_suite.py`,
`scripts/fortress_eval.py`, `scripts/netcheck.sh`, `scripts/strength.sh`, `scripts/registry.py`.

## Data, teachers and licences

- Training positions: Lc0 public training data (ODbL); any derived dataset that is published stays ODbL.
- Teachers: Lc0 (MCTS W/D/L in its data), Stockfish 19 (pinned binary, SHA-256 in the dataset manifest) as a
  relabeling teacher, game results, and Syzygy tablebases for ≤ 5 pieces. Blend 0.6 Lc0 + 0.25 Stockfish + 0.15 result.
- Tablebase probing: Pyrrhic (MIT), vendored in `third_party/pyrrhic`.
- Published nets are listed with their lineage in `nets/registry.csv`.
