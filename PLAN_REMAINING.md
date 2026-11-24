# Bulwark: plan for the remaining work (Project B, defensive NNUE chess engine)

> Source of truth for scope and statistics: `project_B_nnue_chess_engine.md`. Frozen formats: `docs/interfaces.md`, `docs/pipeline.md`. Written 2026-10-09.


## Context

Bulwark is a bitboard C++ engine (UCI, Chess960, PVS) meant to become a *defensive specialist* with a self-trained W/D/L NNUE distilled from Lc0 + Stockfish. The goal is a measurable claim: a draw-preferring utility on a fixed net improves the save rate on held-out unbalanced positions, at a measured strength cost. Everything so far is the foundation (engine + data/training pipeline). None of the headline results exist yet: no trained net, no NNUE in search, no defense study.

## Where we are (evidence)

| Area | State |
|---|---|
| Phase 0-1 movegen/Chess960 | Done. Perft exact on all CPW positions + 241 Chess960 positions (python-chess reference), ASan clean, ~86 Mnps |
| Phase 2 UCI/search/TT/PeSTO/bench | Done except: soak was 400 games (gate says 1000); bench count never recorded; no fastchess Chess960 games yet |
| Phase 3 selectivity ladder (4+0.04, 600-game cap) | **All 9 steps measured**: aspiration +21 (no verdict), RFP +53 (no verdict), NMP +80 (H1), LMR +148 (H1), killers/history +149 (H1), LMP+futility +48 (no verdict), IIR -15 (no verdict, **default off**), improving +8 (no verdict), SEE pruning +56 (H1) |
| D3/D4/D5 (switches) | Implemented. D4 Syzygy (vendored Pyrrhic) verified against python-chess on 400 positions; **no strength SPRT run yet for D3/D4/D5, nor for TMScaling** (the ladder ran with it off) |
| Phase 4 pipeline code | Done: Lc0 v6 converter, Stockfish relabel, manifest, records/splits, loader with resumable cursor, N1/N1-WDL trainer (exact resume tested), export, C++ float+quantized inference. Float inference matches PyTorch to 1.4e-6; quantized within 0.055 logits. 17 pipeline tests + C++ suite green |
| Tooling | fastchess built; Stockfish 19 pinned (sha256 in `~/tools/stockfish/SF19.sha256`); Syzygy 3-5 piece (940 MB) + 8moves_v3 on disk |
| Not started | Kaggle pilot/production data, any GPU training, NNUE in search, D1/D2, benchmark sets, N2, MoE, write-up |

Nothing is running in the background; the laptop is free for timed matches.

## Inputs needed from the user (blockers)

1. Kaggle account/API token (CPU sessions + T4x2 GPU) and a Hugging Face write token (private model repo for checkpoints).
2. Confirm the Lc0 source run (`test91`, input format 1, v6) and that ~100+ GB of Kaggle-side download is acceptable (nothing is downloaded to the laptop).
3. Permission to use the optional server; otherwise everything stays on laptop + Kaggle.
4. Laptop discipline for timed matches (plugged in, idle) during overnight runs.

## Work packages (in dependency order)

### WP-A. Close Phase 3 (laptop, mostly waiting on matches)
Code/doc work can overlap with WP-B/C because only the timed runs need an idle laptop.
1. Set final defaults from the ladder (IIR off, others on); record `Bench: <nodes>` in every later commit (`./build/bulwark bench`).
2. Timed one-switch SPRTs via `scripts/feature_sprt.sh`: TMScaling, D3_Rule50Scale, D5_StalemateGuard, D4_Syzygy (all `[-5,5]` style bounds, 4+0.04, cap ~600 games), then IIR re-test (1200 games) to decide its default.
3. Combined confirmation: final build vs Phase 2 build (`engines/bulwark-p2`), confirm the sum of the ladder.
4. Robustness gates from Phase 2: 1000-game legality/time-loss soak at 8+0.08; Chess960/DFRC fastchess run (`-variant fischerandom`, DFRC book) with zero illegal moves; UCI protocol tests (position/go/stop/ucinewgame, `UCI_Chess960` castling spelling).
5. fastchess scripts for every B.6 table row (net-vs-net fixed nodes, save-rate, fortress/tablebase suites, strength benchmark) + a results-log format (`results/registry.csv`).
6. Strength benchmark vs Stockfish 19 `UCI_Elo` (3 levels around the estimate, 60+0.6, ~200 games each, overnight, Ordo for the estimate).
7. Write `README.md` with the Elo-per-feature table (this plan's table), perft Mnps, bench, benchmark result.
**Exit:** all SPRT rows filled, soak/960 clean, README. This is the "complete HCE engine" milestone (behind the sprint's Day 3 gate).

### WP-B. Finish Phase 4: data pilot, dataset, N1/N1-WDL (Kaggle)
1. Pilot (`kaggle/pilot.sh`, 3 tars, `RELABEL_LIMIT=20000`): measure download MB/s, convert pos/s, filter acceptance, Stockfish relabels/s/vCPU and actual nps, label sanity, bytes/record, kill-and-resume on Kaggle, GPU examples/s, current Kaggle limits. Local reference: ~20k pos/s convert, 64% acceptance, ~89 relabels/s/worker under load.
2. Add the missing data checks before scaling: tablebase-override unit test against the Lichess tablebase API (cursed/blessed categories); a sanity net trained on pilot data; dedup/split leakage check on game ids.
3. Set node budget / dataset size from measurements (target 20-50M positions for N1/N1-WDL; one 12h round on 5 CPU sessions; fall back to 1k nodes or a subset if relabel throughput is short). Run production sessions with disjoint tar slices; consolidate shards into a private Kaggle dataset `nnue-data`; `pipeline/make_training.py` for train/val/test.
4. Train N1 (scalar) and N1-WDL (3 logits), T4x2, ~1-3 h each, deadline-aware, mirrored to HF. Cross-check N1 against bullet (needs a small `to_bulletformat` converter, not written yet).
5. Export `.nnue` + `.nnuef` + `ref-*.jsonl`; run `verifynet` on both.
**Exit:** pilot report, data checks pass, N1 and N1-WDL trained, resume test passes on Kaggle.

### WP-C. Phase 5: NNUE in the engine
1. Load a net via `EvalFile`; evaluation dispatch in `src/eval.cpp` (HCE vs NNUE switch).
2. Incremental accumulator: per-ply accumulator stack in `Searcher` (copy-make compatible), lazy add/sub updates from move deltas, refresh on king-bucket change later; AVX2 quantized kernels in `src/nnue.cpp`, verified against the scalar path (reuse `NNUE::verify`) and PyTorch.
3. WDL head to search scale: neutral mapping `u = 400*logit(W + D/2)` (the D1 hook), scalar head `u = 400*out`; check the eval-scale vs margin coupling and retune RFP/futility/LMR margins after the switch.
4. `bench` with net; fixed-node SPRT NNUE vs HCE (e.g. `nodes=25000`), then timed milestone SPRT; strength benchmark again.
**Exit:** NNUE beats HCE at fixed nodes and fixed time; quantized == float within tolerance; milestone in the Elo table.

### WP-D. Phase 6: the defense study (the core claim)
1. Implement D1 (utility `U = W + d*D`, internal scale, opponent-node weight `1-d`) and D2 (situational draw value, `DrawValue`/`DefenseMode`), with TT handling for `d` (clear or utility-tag entries) and 50-move interaction. Everything stays a UCI switch; comparisons flip exactly one switch (or D1+D2).
2. Benchmark sets: download `UHO_Lichess_4852_v1.epd`; carve disjoint calibration (200) / dev (300) / test (500) + sealed (500) by source game; Stockfish decides the worse side; frozen files committed with hashes.
3. Calibrate `N_sf` on N1-WDL at neutral settings (save rate ~30%); fixed-node save-rate harness (no adjudication, 3-5 piece tables only), paired bootstrap/exact test clustered by source game, W/D/L transition table.
4. Suites: tablebase defense (~150 verified draws via Lichess API, dev/test halves) and fortress suite (~50, CPW + Guid & Bratko + drawn imbalanced game positions).
5. Experiments: N1-WDL vs N1 at neutral utility; D1+D2 on vs off (tune on dev, report once on test); non-regression SPRT `[-5,0]`; D3-D5 ablations one at a time; probability quality (teacher log loss/Brier, outcome calibration with ECE and a reliability diagram).
**Exit:** defense result with 95% CIs and the measured strength trade-off. Fast track ends here plus the write-up.

### WP-E. Phase 7: N2 capacity (stretch)
Mirrored 8 king buckets, L1 512, `2N -> 16 -> 32 -> 3`, Finny tables; trainer feature extension and export format v2 (arch id 3); C++ AVX2 + refresh logic + parity; SPRT vs N1-WDL; re-run the frozen D1+D2 comparison on the test set. GPU ~6-8 h.

### WP-F. Phase 8: MoE (stretch)
Variants A dense / B 8 piece-count stacks / C learned top-1 router (+ optional top-2), Switch loss + load balancing, sparse upcycling from N2; int8 router in C++ with parity; equal-active-compute table (teacher agreement, fixed-node/time Elo, NPS, defense benchmark once, flip rate, utilization). A clean negative result is acceptable.

### WP-G. Phase 9: final runs and write-up
Final strength benchmark; reporting runs on the sealed 500; results page + README with real numbers and per-feature table; disclose teachers/licenses (Lc0 data ODbL, Stockfish as teacher with version/hash, Pyrrhic MIT); publish nets with lineage in `nets/registry.csv`.

## Sequencing and effort

Order: A and B start together (A needs the laptop, B needs Kaggle and tokens); C coding can start as soon as a trained or even a random-weight test net exists (parity harness is ready); D coding (D1/D2, set carving, harnesses) overlaps with C's matches; E/F only if the D gate is green on schedule.
Rough effort at 12 h/day with AI assistance: A 1 day (+1 overnight), B 1-2 days (limited by Kaggle), C 1-2 days, D 2 days, E 1 day, F 1 day, G 1 day. Core (A-D + write-up) ~6-7 days; with E-F ~9 days. GPU budget ~18-22 of 30 Kaggle h/week for N1/N1-WDL/N2/MoE.

## Kill switches / cuts (keep from the master plan)

Relabeling too slow for 50M -> 1k nodes or train on one round. Learned router unstable -> report A vs B only. N2 not beating N1-WDL at fixed nodes -> skip MoE, finish N2 later. Cut order: MoE -> N2 -> D6-D8 -> ladder 13-15. Never cut: perft gate, testing infrastructure, data pilot, WDL head, held-out defense benchmark.

## Known risks / debt to track

- Ladder deviations from the plan: 4+0.04 and a 600-game cap, so several steps have no SPRT verdict; they stay on because the point estimates are positive.
- Quantized-vs-float error is ~5 cp mean; QAT or scaling tweaks if it costs Elo at Phase 5.
- Position is ~700 bytes with copy-make; revisit if NPS suffers once accumulators are added.
- Lc0 data mixes in Chess960 games; castling rights are dropped when not on standard squares (documented). Ep is not in Lc0 format 1.
- Laptop noise: keep concurrency 5, plugged in, idle; prefer fixed-node runs on Kaggle for net comparisons.
- Unfinished small items: `Threads` option is fixed at 1; no `searchmoves`/ponder; bullet cross-check converter; TB root filter only for <=5 pieces.

## Verification (per package)

- WP-A: `build/run_tests tests/data --full`, `scripts/soak.sh`, `scripts/feature_sprt.sh`, strength-benchmark logs in `results/`; README table matches the logs.
- WP-B: `python3 -m unittest discover -s tests -p test_pipeline.py` (with `LC0_TAR` for the real-data check), pilot report JSON, resume test on Kaggle, `verifynet` float < 1e-4 and quantized < ~0.15.
- WP-C: `bulwark bench` deterministic; SPRT NNUE vs HCE passes; scalar vs AVX2 vs float parity script.
- WP-D: frozen set hashes; calibration gives ~30% save rate; each reported comparison runs once on test with CI, hypotheses, LLR and game counts.
- WP-E/F: parity tests per new net format; equal-compute table reproducible from `results/registry.csv`.
