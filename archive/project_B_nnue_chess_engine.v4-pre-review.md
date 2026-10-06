# Project B: Defensive NNUE Chess Engine (final plan)

## Context for the new chat

- **Other work:** a separate project (Project A, planned in its own file) shares the weekly time budget.
- **Sequencing:** about 25 weeks in total for A and B at 10-15 hours per week, overlapped. **B alone is about 285-405 focused hours** (fast track about 240-345; complete HCE engine at Phase 3 about 105-155). The "18 weeks" in B.10 assumes about 20 h/week on B. At a shared 10-15 h/week, B does not fully fit alongside A (see the effort table in B.10). The engine is complete and usable on its own after **Phase 3**, the fallback if time runs short.
- **Dev environment:** WSL2. Keep source, binaries and data on the Linux filesystem (`~/...`), not `/mnt/c`.
- **History:** earlier versions of this plan are in `archive/`. The decision log at the end explains how the plan got here.

### Locked decisions
1. **Engine:** a bitboard C++ engine with UCI, standard chess and **Chess960/DFRC**, PVS alpha-beta search, and a self-trained NNUE.
2. **Identity: a defensive specialist.** It recognizes fortresses and steers worse positions toward draws (repetition, stalemate, the 50-move rule, tablebase draws). The claim is proven with a defense benchmark (B.6).
3. **Evaluation:** an NNUE with a **W/D/L (win/draw/loss) output**. The net is big in capacity but sparse in compute: king-bucketed input layer, plus a **mixture-of-experts** experiment after the accumulator.
4. **Labels:** distilled from teachers, with blend weights **0.6 Lc0 + 0.25 Stockfish + 0.15 game result**.
   - Lc0: public self-play data (ODbL licence) with MCTS W/D/L per position.
   - Stockfish: relabeling at about 2k nodes per position on free Kaggle CPU sessions.
   - A from-scratch self-play track is optional.
5. **Opponent and rating anchor: Stockfish**, pinned to one release.
   - Fixed node limits for the save-rate benchmark.
   - `UCI_LimitStrength` / `UCI_Elo` for the rating anchor (calibrated by Stockfish at 60+0.6 against CCRL 40/4).
   - Time controls are locked in B.6.
6. **Low compute:** everything runs on the laptop plus free Kaggle sessions. A server exists but is optional, only if permitted.

### Hardware
| Machine | Specs | Used for |
|---|---|---|
| **Laptop** | AMD Ryzen 5 5500U (Zen 2, 15 W), 6 cores / 12 threads, AVX2 + BMI2, no AVX-512, **slow PEXT**; WSL2 sees 7.4 GiB RAM; about 148 GB free disk | Development, perft, timed SPRT matches (5 concurrent games), anchor gauntlets, reporting benchmarks |
| **Kaggle CPU** (free) | At most 5 concurrent batch sessions, 12 h each, about 4 vCPUs each; no weekly CPU quota; internet enabled (the account is phone-verified) | Lc0 download and conversion, Stockfish relabeling, fixed-node matches |
| **Kaggle GPU** (free) | P100 or 2x T4, about 30 GPU-h/week, 12 h sessions | Training |
| **Colab** (free) | T4, up to about 12 h per session; GPU limits are unpublished and dynamic (plan on a few hours per day when available) | Second GPU source for training (not for unattended CPU jobs) |
| **Hugging Face** (free) | 100 GB private storage (public is best-effort); Spaces CPU Basic (2 vCPU, 16 GB) for demos; ZeroGPU only 3.5 min/day; Jobs are paid | **Storage, not compute:** the cross-platform home for checkpoints and exported nets, reachable from Kaggle and Colab. Optional: a free CPU Space for a demo or a Lichess bot. |
| Server (optional) | Xeon Silver 4314, 16 cores / 32 threads, 251 GB RAM, AVX-512 + VNNI | Any CPU job, faster, **only if permitted**. No phase depends on it. |

---

## B.1 Goal and success criteria

A UCI-compliant (standard + Chess960), bitboard C++ engine with a tuned PVS search and a WDL NNUE distilled from Lc0 and Stockfish. It is specialized for defense, every change is SPRT-validated, and strength is reported against Stockfish rating anchors.

**What "defensive monster" means, measurably** (B.6):
1. **Save rate:** from unbalanced openings, the engine plays only the worse side against frozen Stockfish settings. Report (draws + wins) / games for the defensive build against the non-defensive baseline, with a paired 95% CI.
2. **Fortress suite:** share of curated fortress positions where the engine (a) evaluates near a draw within a fixed budget, and (b) holds the draw in play.
3. **Tablebase defense:** 6-7 piece positions that are verified draws despite a material deficit, played with only 5-piece tablebases loaded: the hold rate. In lost positions: plies survived compared with DTZ-optimal resistance.
4. **No collapse in overall strength:** non-regression SPRT against the non-defensive build on a balanced book.
5. **WDL calibration:** predicted vs. actual draw frequency on held-out games (reliability diagram, expected calibration error).

| Milestone | Measurable outcome | Rough Elo prior (replace it with the anchor-gauntlet result) |
|---|---|---|
| Phase 1: movegen | Perft exact on the 6 CPW positions + an EPD suite + the Chess960 suite; perft Mnps | n/a |
| Phase 3: HCE engine + tablebases | Elo-per-feature SPRT table, anchor gauntlet | ~2200-2600 |
| Phase 5: first NNUE | +Y Elo over HCE, at fixed time **and** fixed nodes | ~2700-3100 |
| Phase 6: defense | Save rate +X points over baseline at no Elo loss; fortress and tablebase hold rates; calibration | n/a (style metric) |
| Phase 7: MoE | Learned router vs piece-count stacks vs dense, at equal active compute | Report Δ, including negative results |

On stalemate tricks: search already scores stalemate as a draw. Against **engines**, what saves games is accurate draw prediction plus a search that steers toward draws. Traps that rely on an opponent blundering only pay off against **humans**, so they are a stretch goal for a Lichess bot (B.4, D8).

## B.2 Mental model: three layers and where they couple

1. **Board representation and move generation:** correctness first, speed second. A bug here silently poisons search, testing and training data.
2. **Search:** makes the static eval more accurate by looking ahead. Modern strength comes from *selectivity* driven by good move ordering. **Style** lives here, as a utility function plus draw handling.
3. **Evaluation:** `eval(position)` returns W/D/L probabilities. Search turns them into one scalar using the engine's draw value.

Where the layers couple:
- **Eval scale ↔ pruning margins:** all margins are in centipawns, so changing the eval or the draw value means retuning them.
- **Eval speed ↔ depth:** report fixed-node and fixed-time results separately.
- **Movegen correctness ↔ data quality:** this is what the perft gate protects.
- **TT ↔ path-dependence:** mate-score ply adjustment, repetitions, the 50-move counter and the draw value `d`. TT scores depend on `d`, so bump the TT age (or clear) when `d` changes between moves.
- **Data neutrality:** labels come from neutral teachers, and any own data generation runs at `d = 0.5`. Style belongs in play-time search, or the net learns a biased draw probability.

Win probability: `sigmoid(cp/400)`. It maps back as `cp = 400*logit(p)`.

---

## B.3 Board representation (decided)

Modern engines are **hybrid**: bitboards for move generation, attacks and eval, plus a mailbox for O(1) "what's on square x".

| Decision | Choice | Notes |
|---|---|---|
| Square mapping | **LERF**: a1 = 0 ... h8 = 63, north = `<< 8` | `sq ^ 56` flips ranks (black's perspective in PSTs and NNUE). |
| Piece storage | 6 piece-type bitboards + 2 color bitboards + `Piece mailbox[64]` | One AND per "white knights" query. |
| Sliding attacks | **Fancy magic bitboards**: about 800 KiB rooks + 38 KiB bishops. Find your own magics (random sparse candidates, collision-checked). | A PEXT build flag exists but is **off by default**. PEXT is microcoded and very slow on the Zen 2 laptop, and fast only on Intel Haswell+ / Zen 3+. |
| Other tables | Knight, king and pawn attacks; `between[64][64]`, `line[64][64]` | Pins, check blocks and castling paths each become one AND. |
| Move encoding | 16-bit: from (6), to (6), flags (4) | **Castling is encoded as king-captures-own-rook** (needed for Chess960). |
| Make/unmake | **Copy-make** into a ply-indexed stack | NNUE accumulators are per-ply anyway. |
| Legality | Pseudo-legal staged generation + `is_legal(move)` with precomputed `checkers` and `pinned` bitboards; a separate evasion generator | Fits staged ordering. |
| Zobrist | 12x64 + side + 16 castling-mask keys + 8 en passant file keys; **hash the en passant file only if a capture is possible**; pawn and non-pawn sub-keys | Debug assert: incremental key == recomputed key. |
| Draws | Repetition (scan back to the last irreversible move, step 2), 50-move rule (mate on move 100 takes precedence), insufficient material | Single repetition = draw inside the tree; threefold against game history. |

### Chess960 / DFRC
- Store `castling_rook_square[4]`. The castling-rights mask table covers the king square and each castling-rook square.
- **Final squares are fixed:** king to g1/c1 (g8/c8), rook to f1/d1 (f8/d8). The king may not move at all, and the rook may already be on its destination.
- **Legality:**
  - Every square between the king and its destination, and between the rook and its destination, is empty except for that king and that rook.
  - The king is not in check, and every square it passes through, destination included, is unattacked.
  - **Compute those attacks with the castling rook removed from occupancy.** Example: a rook on b1 shields the king from a queen on a1; after queenside castling the king on c1 would be in check.
- **Make:** remove the king and rook first, then place both. They can swap or overlap squares.
- **FEN:** parse `KQkq` (X-FEN, meaning the outermost rook) and **Shredder-FEN** rook-file letters (`HAha`; uppercase = white). Print Shredder-FEN in 960 mode.
- **UCI:** `option name UCI_Chess960 type check default false`. In 960 mode, send and receive castling as king-to-rook (`e1h1`). In standard mode use `e1g1`.
- **Perft:** the CPW *Chess960 Perft Results* page has 850 positions with depth 1-6 counts. Run a sample in CI and the full set before Phase 2.

### Perft gate (Phase 1 exit)

| Position | FEN | Depth: nodes |
|---|---|---|
| Start | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` | 5: 4,865,609 / 6: 119,060,324 |
| Kiwipete | `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -` | 4: 4,085,603 / 5: 193,690,690 |
| 3 | `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1` | 6: 11,030,083 |
| 4 (and its mirror) | `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1` | 5: 15,833,292 |
| 5 | `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8` | 4: 2,103,487 |
| 6 | `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10` | 4: 3,894,594 |
| 960 #1 | `bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9` | 6: 227,689,589 |

Plus a 100+ position EPD suite and the 960 suite. Use `perft divide` against a reference engine to bisect mismatches. Report Mnps with and without bulk counting.

---

## B.4 Search

### Search techniques: what exists, and what to use

| Family | Examples | Fit for a CPU engine with a value-only NNUE | Verdict |
|---|---|---|---|
| Minimax / negamax | Textbook | Exponential, no pruning | Teaching only |
| **Alpha-beta + PVS (NegaScout)** | Stockfish, Ethereal, every top NNUE engine | Tactically sharp, uses a cheap value eval, deep with good ordering | **Core algorithm** |
| MTD(f) | Plaat's MTD; some older engines | Chains of null-window searches that lean hard on the TT; interacts badly with LMR and singular extensions; no modern top engine uses it | No |
| Best-first: SSS*, B*, conspiracy numbers | Historical / research | Memory-heavy, superseded | No |
| Proof-number search | Mate and endgame solvers | Proves binary outcomes; impractical for fortresses with the 50-move rule | Not in the engine |
| MCTS / PUCT | AlphaZero, Lc0 | Needs a **policy** head and GPU-sized nets; weak tactically at CPU speed | No (borrow its WDL idea only) |
| MCTS/alpha-beta hybrids | Komodo MCTS mode, Scorpio | Large complexity for uncertain gain | No |
| Parallel: Lazy SMP vs YBWC / ABDADA | Stockfish switched to Lazy SMP in 2016 | Simple and scales well to about 8+ cores | Stretch only |

**Recommendation:** PVS alpha-beta with the selectivity ladder below, plus the defensive module (D1-D8). Spend the novelty budget on evaluation, draw-awareness and the MoE experiment.

### Core framework (Phase 2; required before SPRT is meaningful)
- Negamax, fail-soft alpha-beta, **PVS** with templated `PV` / `NonPV` nodes. Iterative deepening, triangular PV table.
- **Quiescence search:** stand-pat; captures and promotions in MVV-LVA order; all evasions when in check; TT probe.
- **TT:** buckets, depth-preferred + aging, ply-adjusted mate scores, no cutoffs at PV nodes, prefetch.
- Search stack per ply: static eval, killers, current move, excluded move, continuation-history pointer.
- Repetition and 50-move draws, mate distance pruning, check extension.

### Move ordering (staged move picker)
TT move, then good captures (SEE >= 0, MVV + capture history), then 2 killers, then the countermove, then quiets by butterfly + continuation history, then bad captures. SEE is a threshold test `see_ge(move, t)` (swap algorithm with x-rays).

### Selectivity ladder: one SPRT per step, in order (CPW *Search Progression*)

| # | Feature | Gist / typical start (tune later) |
|---|---|---|
| 1 | Aspiration windows | From depth ~4, ±~25 cp, widened geometrically on fail. |
| 2 | Reverse futility pruning | Non-PV, not in check, depth <= ~8: `eval - 80*depth >= beta`, then return eval. |
| 3 | Null move pruning | Non-PV, not in check, has non-pawn material, `eval >= beta`; `R = 3 + depth/3` (+ eval term). |
| 4 | Late move reductions | `R = 0.78 + ln(d)*ln(m)/2.47` for quiets; adjust for PV, cut node, history; re-search on fail high. |
| 5 | Butterfly history + killers | Gravity update `h += b - h*abs(b)/MAX`; malus for failed quiets. |
| 6 | Late move pruning + futility | LMP after `3 + d²` quiets; futility `eval + margin(d) <= alpha`. |
| 7 | IIR | No TT move at depth >= ~4: reduce depth by 1. |
| 8 | Improving | eval > eval two plies ago; feeds RFP, LMP, LMR. |
| 9 | SEE pruning | Main search `!see_ge(m, -k*d)`; qsearch prunes SEE < 0. |
| 10 | Continuation history | 1-ply and 2-ply. |
| 11 | Capture history | [piece][to][captured]. |
| 12 | History pruning | Low-depth quiets with very negative history. |
| 13 | Singular extensions + multicut + double/negative extensions | `beta_s = ttScore - k*depth`, half-depth excluded-move search. |
| 14 | Correction history | Pawn, material and non-pawn keyed EMA of `search - static`; also a "position complexity" signal. |
| 15 | Qsearch futility / delta | Skip captures that cannot reach alpha. |

Retune the constants after the NNUE switch and after enabling the draw-value utility.

### Defensive module (the specialty)
Measured on the defense benchmark (B.6) **and** a non-regression SPRT for overall strength.

| # | Technique | What it does | Notes |
|---|---|---|---|
| D1 | **WDL eval → utility** | The net gives W, D, L for the side to move. The engine optimizes `U = W + d*D` from its own perspective. Search uses `cp = 400*logit(U)`. At opponent-to-move nodes the draw weight is `1 - d`, which keeps negamax consistent. | `d = 0.5` is neutral (expected score). |
| D2 | **Situational draw value** | At the start of each move, set `d` from the previous root WDL: neutral by default; about 0.6-0.7 when root P(loss) is high ("defense mode") or against a stronger opponent (UCI options `DrawValue`, `DefenseMode`). In-tree draws score `400*logit(d)` for the engine. | Never use a large `d` globally: `d = 0.6` values a draw like +162 cp, so the engine would chase draws from equal positions. |
| D3 | **50-move awareness** | Scale the eval toward the draw score as the halfmove clock grows, Stockfish-style (`v * (212 - rule50) / 212`). | Ablation: post-hoc scaling vs. `rule50` as a net input. |
| D4 | **Syzygy tablebases** (Fathom, MIT, or Pyrrhic) | 3-5 piece tables (about 939 MiB). Probe WDL in search (not qsearch) when pieces <= 5, `rule50 == 0` and no castling rights. At the root, keep only drawing or winning moves by DTZ. When lost, play the **max-DTZ** move, and treat blessed losses (lost, but saved by the 50-move rule) as draws. | 6-piece tables (149 GiB) are optional. |
| D5 | **Stalemate guard in qsearch** | In low-material positions, check whether the side to move has any legal move before standing pat; if not, score a draw. | Cheap; gate by a material threshold. |
| D6 | **Perpetual-check seeking** | When the defender is far below alpha, search quiet checks at the first qsearch ply and extend checks. | SPRT carefully; it inflates trees. |
| D7 | **Fortress / no-progress detection** (Guid & Bratko 2012) | At the root: if \|score\| is large but flat across many iterations and the PV has no zeroing move, scale toward a draw. | Experimental; judged on the fortress suite. |
| D8 | **Swindle mode** (humans only, stretch) | Among moves with nearly equal `U`, prefer ones where many replies lead to draws. | Lichess bot only. |

### Search limits: there is no fixed depth in real play
Iterative deepening searches depth 1, 2, 3, ... until a limit stops it.

| Mode | Limit | Used for |
|---|---|---|
| `go wtime/btime/winc/binc` | **Time** (soft/hard, below) | SPRT matches, anchor gauntlets, reporting benchmarks |
| `go movetime T` | Fixed time per move | Analysis |
| `go nodes N` | Fixed node count | Fixed-node matches, the routine save-rate benchmark |
| `go depth D` | Fixed depth | Debugging, `bench` |
| `go infinite` | Until `stop` | GUI analysis |

`MAX_PLY` (about 128-256) is only a safety cap that sizes the per-ply arrays. Real depth varies per line (reductions, extensions, qsearch → `seldepth`). If the hard limit fires mid-iteration, play the best move from the last completed iteration (or the current one's, if it was already fully searched).

### Time management
- `soft = remaining/20 + inc/2`. The hard limit is a few times soft, capped well below remaining time, minus `Move Overhead`.
- Check the soft limit between iterations and the hard limit every N nodes.
- Upgrades: node-based scaling, best-move stability, eval stability. **Defense extra:** spend more time when root P(loss) jumps.

### Parallel search
Lazy SMP is a stretch goal. Testing runs many single-threaded games in parallel instead.

---

## B.5 Handcrafted evaluation: small on purpose

Its only job is to be the Elo baseline for the NNUE delta. Keep it minimal and timeboxed.
- Tapered PeSTO material + PSTs, phase N = B = 1, R = 2, Q = 4 (total 24): `(mg*phase + eg*(24 - phase)) / 24`.
- **Drawishness scaling** (opposite-colored bishops, rook pawn + wrong bishop, low-material endings), a cheap first taste of the defense identity.
- Optional terms, each SPRT'd: mobility, passed pawns, king safety. Optional Texel tuning on converted Lc0 positions with results.
- **Symmetry test:** `eval(pos) == eval(color_flip(pos))`.

---

## B.6 Testing and measurement

- **Git from day 1.** Every functional commit carries `Bench: <nodes>`. `bench` runs a fixed-depth search over about 50 FENs (standard and 960).
- **Correctness suite:** perft (standard + 960), FEN round-trip (X-FEN and Shredder-FEN), Zobrist check, SEE tests, eval symmetry, mate-in-N, tablebase probe sanity, float-vs-quantized net check. Debug builds use asserts and `-fsanitize=address,undefined`.
- **Match runner:** fastchess (pentanomial SPRT), single-threaded engines, Hash 16 MB. Each `-engine` takes its own limit (`tc=` or `nodes=`), so your engine can play on time while Stockfish is limited by nodes.
- **Laptop discipline for timed matches:** 5 concurrent games (one per physical core, leaving one for WSL and the OS). Plugged in, best-performance power mode, no builds or other heavy jobs during a test. Fixed-node matches can use up to 10 concurrent games.
- **SPRT bounds** (alpha = beta = 0.05, so LLR ±2.94): `[0, 10]` / `[-10, 0]` while the engine is weak; `[0, 5]` / `[-5, 0]` later. Large early gains pass in a few hundred to about 1,000 games, roughly an hour at STC.
- **Books:** `8moves_v3.pgn` (balanced) for SPRTs and anchors; `UHO_Lichess_4852_v1.epd` (unbalanced) for the save-rate benchmark and for SPRTs once draw rates climb; a DFRC book for 960 tests (`-variant fischerandom`).

### Time controls and opponents (locked)

| Purpose | Your engine | Opponent | Book | Where | Adjudication |
|---|---|---|---|---|---|
| **SPRT, every change** (STC) | `tc=8+0.08` | Previous build, `tc=8+0.08` | 8moves_v3 | Laptop, 5 concurrent | `-resign movecount=3 score=600`, `-draw movenumber=40 movecount=8 score=10` |
| **SPRT, depth-sensitive changes** (LTC) | `tc=40+0.4` | Previous build, `tc=40+0.4` | 8moves_v3 | Laptop, overnight | Same |
| **Net-vs-net checks** | `nodes=25000` | Previous net, `nodes=25000` | UHO | Kaggle CPU or laptop | Same |
| **Save rate, routine** | `nodes=50000` | **Stockfish `nodes=N_sf`** | UHO; you play the worse side | Kaggle CPU or laptop | **Syzygy only (`-tb`); no resign or draw adjudication** |
| **Save rate, reporting** | `tc=8+0.08` | **Stockfish `nodes=N_sf_tc`** | Same | Laptop | Same |
| **Fortress and tablebase suites** | `tc=8+0.08` | **Stockfish `nodes=N_sf_tc`** | Suite positions | Laptop | Same |
| **Rating anchor** (CCRL-equivalent) | `tc=60+0.6` | **Stockfish `UCI_LimitStrength=true`, `UCI_Elo` = your estimate −200 / ±0 / +200**, `tc=60+0.6` | 8moves_v3 | Laptop, overnight (~600-900 games) | Standard |

**Stockfish settings, everywhere:** one pinned release (record the version and binary SHA-256), `Threads=1`, `Hash=16`, `SyzygyPath` pointing at the same 3-5 piece tables your engine uses.

**Why two different Stockfish limits:**
- **Node limits for the benchmark.** They give accurate play that is deterministic and independent of CPU speed, so the benchmark measures "can it hold against accurate pressure".
- **`UCI_Elo` for the rating anchor.** Stockfish calibrated that mode at 60+0.6 against CCRL 40/4, which is why the anchor runs at exactly that time control. It weakens Stockfish with deliberate random errors, which is fine for rating but wrong for the defense benchmark.
- The anchor range tops out around `UCI_Elo` 3190.
- Compute the rating from the three anchor results with Ordo, holding the anchors fixed.

### Save-rate protocol (frozen at the start of Phase 6)
1. **Positions:** 500-1,000 UHO positions. A quick Stockfish eval decides the worse side, which your engine always plays. Every build plays the same positions.
2. **Calibration (once):** run the Phase 5 baseline (NNUE, defense features off) against Stockfish at nodes in {5k, 10k, 25k, 50k, 100k}, at both your routine setting (`nodes=50000`) and your reporting setting (`tc=8+0.08`). Pick `N_sf` and `N_sf_tc` where the baseline saves about **30%**. Too strong gives about 0% for every build (floor); too weak gives about 100% (ceiling). Freeze both values.
3. **Runs:** use routine runs for every defense change and reporting runs at milestones. Fixed-time numbers matter because defense features that cost speed only show their real cost there.
4. **Statistics:** save rate = (draws + wins) / games. Compare builds paired by position. At about 0.3, 500 positions give a 95% CI of roughly ±6 points on the difference, and 1,000 give roughly ±4. Report the CI and the per-position W/D/L transition table.
5. **Optional second opponent:** a different open-source engine of similar strength, to check that the defense isn't tuned only against Stockfish's style.

### Defense benchmark

| Test | Setup | Metric |
|---|---|---|
| Save rate | Protocol above | Save rate with paired 95% CI |
| Fortress suite | Curated from the CPW *Fortress* examples, Guid & Bratko (2012), and drawn-but-imbalanced positions from match games | % recognized (P(draw) > 0.6 or \|eval\| < 50 cp within the budget); % held in play |
| Tablebase defense | 6-7 piece positions drawn despite a material deficit, verified with the **Lichess tablebase API** (`tablebase.lichess.ovh/standard?fen=...`, 7-piece coverage; cache and throttle). Your engine has 5-piece tables only. | Hold rate; for lost positions, plies survived vs DTZ |
| Calibration | Held-out games | Reliability diagram, ECE for P(draw) |
| Strength guard | SPRT `[-5, 0]` vs the non-defensive build, STC, balanced book | Pass/fail plus Elo ± error |

No standard public fortress suite is assumed. Building and publishing one is itself a contribution.

---

## B.7 Labels: multi-teacher WDL distillation

### Label sources per position
| Source | What it provides | Cost | License / disclosure |
|---|---|---|---|
| Game result | W/D/L one-hot. Noisy, but grounded in what actually happened. | Free | n/a |
| **Lc0 public training data** (v6 chunks, `storage.lczero.org/files/training_data/`) | `root_q`, `root_d` (MCTS value and draw probability after search), `best_q/best_d`, game result (`result_q/result_d`), `rule50`, and the board as 104 bit-planes | Download + convert (in Kaggle sessions) | **ODbL**: credit the source; any derived dataset you publish stays ODbL. Disclose Lc0 as a teacher. |
| **Stockfish relabel** at about 2k nodes | Centipawn score, giving expected score and, via Stockfish's WDL model (`UCI_ShowWDL`), W/D/L | About 40M positions per 12 h Kaggle session, about 200M per round of 5 (estimate) | The output generally isn't GPL-covered (its code and nets would be). Disclose it as a teacher. |
| Syzygy (<= 5 pieces) | Exact W/D/L, 50-move aware | About free | n/a |

AlphaZero was never released. Lc0 is its open reproduction, and its public data already stores a searched draw probability for every position.

### Turning labels into targets
- **Lc0 q/d → W/D/L:** `W = (1 + q - d)/2`, `D = d`, `L = (1 - q - d)/2`, from the side to move's view. **Unit-test the perspective** on known won, lost and drawn positions.
- **Blend (locked):** `t = 0.6*WDL_lc0 + 0.25*WDL_sf + 0.15*onehot(result)`. Exact tablebase W/D/L overrides everything at <= 5 pieces, with cursed win and blessed loss both counting as a draw.
- **WDL net loss:** soft cross-entropy `CE(softmax(z), t)` plus an optional expected-score term `alpha*((W + D/2) - E_t)^2`.
- **Scalar N1:** `target = E_t = W_t + D_t/2` of the blend (the result is already inside `t`).
- **Fallback if Stockfish relabeling is unavailable:** the Lichess evaluation DB (CC0, about 416M Stockfish-evaluated positions). Those are different positions, so renormalize the blend weights per sample over the labels present.

### Gotchas
- **Filtering:**
  - Skip early plies, in-check positions, and positions whose best move is a capture or promotion. Decode `best_idx` with Lc0's 1858-move policy table.
  - Skip extreme `|q|`.
- **Size:** a raw v6 record is 8,356 bytes (mostly policy). Stream it inside Kaggle: download a tar, convert, delete.
- **Distribution shift:** Lc0 positions come from Lc0 self-play, not from the positions your search visits. Relabeling your own engine's positions is the optional fix.
- **Shortcut not taken:** Stockfish's Leela-derived binpacks have no draw probability, and nnue-pytorch's reader is GPL-3.

### Data tracks
- **Teacher-distilled:** the main nets; teachers disclosed in the README. Stockfish itself trains on Leela data.
- **Own self-play bootstrap (optional):** the from-scratch experiment, only if spare compute appears.

---

## B.8 NNUE design: big capacity, sparse compute

### Cost model
For L1 width `N`:
- Accumulator update per move: about (2-4 changed features) x N x 2 perspectives int16 adds.
- Post-accumulator layers: about 2N x 16 int8 MACs.
- Doubling N roughly doubles both.
- **Routing (bucket or expert selection) multiplies parameters without multiplying per-eval work.**

### Architecture roadmap
| Net | Input layer | After the accumulator | Output | Phase |
|---|---|---|---|---|
| **N1** | `768 -> 256` (x2 perspectives), SCReLU | none | 1 scalar | 4-5 (pipeline proof; cross-checked with bullet's `simple` example) |
| **N2 "big"** | **Horizontally mirrored, 8-16 king input buckets**, L1 512-1024; Finny tables for bucket-change refreshes | `2N -> 16 -> 32` | **3 (W/D/L)** | 6 |
| **N3 "MoE"** | Same as N2 | **K = 8 expert stacks** `2N -> 16 -> 32 -> 3` + router | 3 | 7 |

N2 with 16 buckets x 768 x 1024 is about 12.6M first-layer parameters (about 25 MB int16), at the per-eval compute of an unbucketed 1024-wide net. King input buckets are already a hard-routed sparse MoE.

### Where experts can and cannot go
- **First layer, learned routing: no.** Every expert's accumulator would need incremental updates on every move (Kx the work).
- **First layer, hard routing by king square: yes.** These are the N2 buckets.
- **After the accumulator, learned routing: yes.** A router costs about `2N x K` int8 MACs, and only one expert stack runs.

### N3 experiment design
- **Variants at equal active compute:** (A) dense single stack; (B) 8 hard stacks by piece count (Stockfish-style); (C) learned top-1 router over 8 stacks; optional (D) learned top-2.
- **Router input:** accumulator output + cheap scalars (phase, material imbalance, `rule50/100`, side to move).
- **Training:**
  - Switch Transformer top-1, with `y = p_i(x) * E_i(x)` in both training and the engine.
  - Load-balancing loss `alpha * K * sum_i(f_i * P_i)`, alpha around 0.01.
  - **Sparse upcycling:** initialize the experts from the trained N2 stack.
- **Quantization:** router int8 on uint8 activations; argmax picks the expert; float softmax over 8 logits.
- **Measure:**
  - Validation loss, ECE, fixed-node and fixed-time Elo, NPS, defense benchmark.
  - Routing flip rate and |Δeval| at flips, expert utilization.
  - Per-expert specialization (phase, material, imbalance, draw rate).
- A clean negative result is still a strong write-up.

### Quantization and inference
- First layer: QA = 255, int16, weights clamped to ±1.98 in training (the SCReLU `v*w` int16 trick).
- Hidden layers: int8 (QB = 64 for N1's output).
- Lazy accumulator updates on a per-ply stack, fused add/sub, AVX2 with a scalar fallback. WDL head: 3 logits, then a float softmax.
- Verify that C++ float matches PyTorch, then that quantized matches float, on 10k positions, before measuring Elo.

### Data scale
N1 needs about 50-100M positions. **Plan on about 300M filtered positions** for N2/N3; rerun the pipeline if validation loss is still falling. The scarce resource is GPU hours, not data.

---

## B.9 Compute plan, data pipeline and training

### Which machine does what
| Workload | Where | Notes |
|---|---|---|
| Development, unit tests, perft | Laptop | |
| Timed SPRTs, anchor gauntlets, reporting benchmarks | Laptop, 5 concurrent games | Plugged in; nothing else heavy running |
| Lc0 download + conversion + Stockfish relabel | **Kaggle CPU batch sessions** (5 in parallel) | Uses Kaggle's bandwidth and disk |
| Fixed-node matches (net checks, routine save rate) | Kaggle CPU sessions or laptop | |
| Training N1-N3 | Kaggle GPU; Colab fallback | 30 GPU-h/week |
| Anything above, faster | Server, **only if permitted** | `nice` everything |

Data generation doesn't run on a GPU: alpha-beta with incremental NNUE evaluates one position at a time inside a branchy search, while GPUs need large batches. With Lc0's data public, no GPU labeling is needed.

### The data pipeline: one Kaggle notebook, run as 5 parallel batch sessions
1. Each session gets a **shard id** and a disjoint slice of the Lc0 file list. Use one recent training run and record which files.
2. **Download** a tar, **convert** the v6 records, **filter**, and write 40-byte records with Lc0 `q/d`, result and `rule50`. Delete the tar. Throttle downloads.
3. **Relabel** with Stockfish at about 2k nodes: one process per vCPU, `Threads=1`, `Hash=16`; use the official x86-64 AVX2 build.
4. **Deadline-aware:** stop taking new tars at about 10.5 h, flush shards, and write `manifest.json` (files, counts, filter statistics, Stockfish version).
5. Shards land in `/kaggle/working/data/` (about 40M positions ≈ 1.6 GB per session).
6. Training notebooks attach these outputs directly as inputs.

**Target:** about 300M positions (about 12 GB) from about 2 rounds of 5 sessions, 1-2 days of wall-clock time, zero laptop CPU. Unit-test the converter on the laptop with a few files first. **Laptop backup:** about 50-100M relabels per night.

### Engine builds
- **Default:** `-march=x86-64-v3` (AVX2 + BMI2), magic bitboards, for the laptop and Kaggle.
- **Optional:** `-march=icelake-server` (AVX-512 + VNNI + PEXT) only if the server is used.

### Record format (about 40 bytes, fixed-size, versioned shard header)
- **Board:** occupancy bitboard (8 B) + 4-bit piece codes (16 B) + side to move / castling / en passant / `rule50` (3 B).
- **Labels:**
  - Stockfish cp (int16; sentinel if absent)
  - Lc0 `q` (int16, scaled) and `d` (uint16, scaled)
  - Own-engine cp (int16, optional)
  - Result + tablebase flag + source id
- **Reserved** bytes for later fields.
- A `to_bulletformat` converter feeds bullet for the N1 cross-check.

### Where everything lives
| Item | Store |
|---|---|
| Training data | Kaggle notebook outputs, attached as training inputs (optionally consolidated into a private Kaggle Dataset `nnue-data`). Only small samples on the laptop. |
| Checkpoints | Kaggle `/kaggle/working/ckpt` (persisted by **committed "Save & Run All" runs**), mirrored to a **private Hugging Face model repo** (`huggingface_hub` upload after each save; token in Kaggle Secrets or Colab secrets). Either platform resumes from the same repo. |
| Exported nets + lineage | Repo `nets/` + `nets/registry.csv` |

Run 12-hour jobs as **committed background runs**. API tokens go in Kaggle Secrets.

### Checkpoint and resume spec
```
ckpt/run-<id>/
  config.yaml          # arch, shard list, blend weights, LR schedule, seed, git sha
  step-<n>.pt          # keep the last 3
  best.pt              # best validation loss
  metrics.jsonl        # step, train/val loss, ECE, LR, positions seen, router utilization
```
- **Checkpoint contents:** model, optimizer, LR scheduler and AMP scaler state; global step; **data cursor** (shard-order seed, shard index, record offset); RNG states (python, numpy, torch, cuda); best validation loss; wall-clock used.
- **When:** every ~20 minutes and at every superbatch end. **Atomic write:** write a temp file, fsync, `os.replace`.
- **Deadline-aware:** `--deadline 11h30m` makes a final checkpoint and exports a net before the kill. There is also a SIGTERM handler.
- **Resume:** `train.py --resume auto` searches `/kaggle/input/*/ckpt`, then Drive, then local `ckpt/`. It refuses a config-hash mismatch unless `--allow-config-change` is passed.
- **Deterministic data order:** the seed fixes the shard permutation, so the cursor alone determines what comes next.
- **Export:** `export.py <ckpt>` writes `net-<sha8>.nnue` (quantized) plus `ref-<sha8>.jsonl` (float outputs on 10k FENs) for the engine's `verifynet` command.
- **Loader:** a C++ (pybind11) decoder to feature indices, since Kaggle's few CPU cores bottleneck Python parsing. Shuffle in RAM (12 GB fits in about 30 GB) or via a shuffle buffer.
- **Reference trainer:** bullet on Kaggle for N1 only, to cross-check the PyTorch trainer.

---

## B.10 Phased plan (about 18 weeks at ~20 h/week; fast track about 15)

| Phase | Weeks | Content | Exit |
|---|---|---|---|
| 0 | 1 (start) | Git, CMake (Release, Debug + sanitizers), test runner | Builds and tests run |
| 1 | 1-3.5 | B.3 including Chess960: magics, staged pseudo-legal generator, copy-make, Zobrist, draws, FEN/X-FEN/Shredder-FEN | Perft gate (standard + 960) |
| 2 | 4-5 | UCI (+ `UCI_Chess960`, Hash, Threads, Move Overhead), ID + PVS + qsearch + TT, PeSTO, basic time management, `bench`, fastchess scripts for every B.6 row | 1000 games with no illegal moves or time losses; first SPRT passes |
| 3 | 5-7 | Selectivity ladder (at least steps 1-9), time-management upgrades, minimal HCE, **D3 + D4 + D5** | **Fallback-complete:** Elo-per-feature table, first Stockfish `UCI_Elo` anchor rating, README |
| 4 | 8-9 | Kaggle data pipeline (converter unit-tested first, Stockfish relabel, about 300M positions); PyTorch trainer + C++ loader + checkpoint/resume; N1 on the locked blend, float inference in C++ | N1 beats HCE at fixed nodes; the resume test (kill, resume, identical loss curve) passes |
| 5 | 10-11 | Quantized AVX2 inference, SPRT vs HCE, retune margins | +Y Elo over HCE; anchor rating |
| 6 | 12-14 | **Defense:** N2 (king buckets + WDL head), D1-D2, D6-D7; **calibrate and freeze `N_sf` and `N_sf_tc`**; full defense benchmark; calibration | Save-rate gain with the strength guard passing; anchor rating. **The fast track ends here, plus a week of write-up.** |
| 7 | 15-17 | **MoE:** N3 variants A/B/C (+ D), upcycling, specialization analysis | Equal-compute comparison table |
| 8 | 18 | Final anchor gauntlet and reporting benchmarks; write-up (results page + blog-style README) | Results page with real numbers |

### Effort estimate (focused hours; first engine, strong C++)
| Milestone | Phase hours | Cumulative hours | Calendar at 10 h/week | Calendar at 20 h/week |
|---|---|---|---|---|
| Movegen + perft gate (P0-1) | 40-60 | 40-60 | 4-6 weeks | 2-3 weeks |
| Playable engine + test infrastructure (P2) | 25-35 | 65-95 | 7-10 | 3-5 |
| **Complete HCE engine (P3)** | 40-60 | 105-155 | 11-16 | 5-8 |
| Data pipeline + N1 (P4) | 45-65 | 150-220 | 15-22 | 8-11 |
| **First NNUE beats HCE (P5)** | 25-35 | 175-255 | 18-26 | 9-13 |
| **Defense specialty + write-up (P6; fast-track end)** | 65-90 | 240-345 | 24-35 | 12-17 |
| MoE experiment (P7) | 35-50 | 275-395 | 28-40 | 14-20 |
| Final gauntlets + write-up update (P8) | ~10 | 285-405 | 29-41 | 15-21 |

- Background compute (SPRTs, Kaggle data rounds, overnight gauntlets, GPU quota for N2/N3) mostly overlaps with coding. Add about 1-2 weeks of calendar slack, mainly for the 30 GPU-h/week quota during Phases 6-7.
- The ranges include normal engine debugging. Expect the low end if movegen and search go cleanly, the high end if subtle TT, mate-score or Chess960 bugs appear.
- **With heavy AI-assisted coding** (an assistant writes most of the code; you review, test and debug): roughly **160-240 h** for the full plan, **135-200 h** for the fast track, **60-90 h** for the Phase 3 engine. Debugging, verification and waiting on tests shrink much less than typing does.

### Compute is not the bottleneck
| Resource | Needed (estimate) | Free supply |
|---|---|---|
| GPU training | N1 1-3 h; blend sweeps 5-10 h; N2 8-20 h; N3 variants 16-32 h. **About 30-65 GPU-hours total**, mostly in Phases 6-7. | Kaggle 30 h/week + Colab (a few hours/day when available): **about 1-1.5 weeks of combined quota** |
| Data pipeline | About 300M positions | 1-2 days on 5 Kaggle CPU sessions |
| Match testing | About 20-30 STC SPRTs (~1 h each), a few LTC overnights, 4 anchor gauntlets (~9 h each), benchmark runs | Laptop overnight + Kaggle CPU for fixed-node runs |

Even with instant coding, the full plan has a **compute floor of about 2-3 weeks**: sequential SPRTs (each feature is tested on top of the previous one) plus the GPU quota for N2/N3. Engineering hours set the real timeline.

### Calendar by commitment (with AI-assisted coding)
| Commitment | Phase 3 engine | Fast track | Full plan |
|---|---|---|---|
| Full-time sprint (~60-80 h/week) | 1-1.5 weeks | 2.5-3.5 weeks | **3-5 weeks** (compute floor included) |
| 20 h/week | 3-5 weeks | 7-10 weeks | 8-12 weeks |
| 10 h/week | 6-9 weeks | 14-20 weeks | 16-24 weeks |

### 1-week sprint (MVP): only full-time, with a reduced scope
Goal: a working, tested NNUE engine with a "defense-lite" result in 7 days (about 70-85 hours). Plan for 10 days, because one perft or NNUE sign bug can eat a day.

| Day | Build | Runs overnight / in the background |
|---|---|---|
| 1 | Board, magics, movegen, make/unmake, Zobrist, FEN; **perft gate (standard chess)** | n/a |
| 2 | UCI, PVS + qsearch + TT + MVV-LVA/killers/history, time management, `bench`, fastchess scripts | First SPRTs; Kaggle CPU converts the **Lichess evaluation DB** (CC0, FEN + Stockfish cp, trivial to parse) into shards |
| 3 | RFP, NMP, LMR, aspiration windows; minimal PyTorch trainer + checkpointing | SPRTs; N1 training on Kaggle GPU (2-3 h) |
| 4 | Float NNUE inference, verified against PyTorch, then quantized + incremental + AVX2 | SPRT: NNUE vs PeSTO |
| 5 | Defense-lite: Fathom Syzygy (D4), rule50 scaling (D3), qsearch stalemate guard (D5), defense-mode draw value | Strength-guard SPRT |
| 6 | Save-rate benchmark lite: 200-300 UHO positions vs Stockfish at fixed nodes (quick calibration) | Anchor gauntlet vs Stockfish `UCI_Elo` (60+0.6) |
| 7 | README: perft, SPRT table, NNUE gain, save-rate delta (rough CI of about ±8-10 points), anchor rating | Buffer for bugs |

**Deferred to after the sprint:** Chess960, the rest of the selectivity ladder, the Lc0 + Stockfish + result blend (the sprint uses Stockfish labels only), the WDL head and king buckets (N2), D6-D7, the fortress and tablebase suites, and MoE.

**Cut order if behind:** (1) Phase 7 shrinks to A vs B, or moves after Project A ships (the fast track); (2) D6-D8; (3) ladder steps 13-15. Lazy SMP is never in scope. **Do not cut:** the perft gate, testing infrastructure, the WDL head, or the defense benchmark.

**Optional, only with spare compute:** own self-play bootstrap and on-distribution relabeling.

## B.11 Experiment menu

| Experiment | Holds fixed | What it shows |
|---|---|---|
| Neutral `d` vs situational `d` (D2) | Net, search | Value of the draw-preference knob |
| WDL head vs scalar net | Data, input layer | Whether explicit draw modeling helps defense and calibration |
| Post-hoc rule50 scaling vs rule50 as a net/router input | Net size | Learned vs hand-coded 50-move awareness |
| Tablebase label rescoring on/off | Data | Effect of exact endgame labels on fortress recognition |
| Dense vs piece-count stacks vs learned router (vs top-2) | Active compute | The MoE question |
| Upcycled vs from-scratch MoE | Data | Data efficiency of upcycling |
| King-bucket count (1 / 4 / 8 / 16) | L1 width | Capacity via routing vs NPS |
| Blend grid around 0.6/0.25/0.15 (N1-sized, cheap) | Positions, architecture | Which teacher matters for strength vs. calibration and defense |
| Lc0-only vs Stockfish-only labels | Positions | MCTS vs alpha-beta as a teacher |
| Own PyTorch trainer vs bullet (N1) | Data, architecture | Trainer correctness |
| *Optional:* own bootstrap, on-distribution relabeling, opening mixes | Varies | Need own data generation |

## B.12 Risks

- **Specialization costs Elo:** a situational `d`, the strength guard, neutral labels.
- **Scope creep:** 285-405 hours (fast track 240-345) against a 25-week budget shared with Project A. Follow the cut order; the engine is complete at Phase 3, and Phase 5 is a strong stopping point.
- **MoE instability:** routing flips cause eval jumps. Measure them; fall back to piece-count stacks.
- **Label bugs in conversion:** perspective, sign or orientation errors silently ruin a net. Use unit tests plus a 10M-position sanity net before scaling.
- **Licensing:** Lc0 data is ODbL (credit it; derived datasets you publish stay ODbL). Reusing nnue-pytorch code makes the trainer GPL-3. State the teachers in the README.
- **Distribution shift:** teacher positions are not your search's positions. Watch for gaps; relabeling own positions is the optional fix.
- **Anchor bias:** one anchor family (Stockfish `UCI_Elo`), calibrated at 60+0.6 against CCRL 40/4, weakened by random errors. Report the rating as an estimate with error bars, not a CCRL listing.
- **Kaggle dependence:** session caps, committed-run persistence, 30 GPU-h/week. Deadline-aware checkpoints and deterministic resume; the laptop as a slower backup.
- **Laptop noise:** a 15 W chip throttles, and WSL shares the machine with Windows. Use 5 concurrency, plug in, keep it idle during timed tests, and rely on fixed-node runs where possible.
- **Movegen and 960 bugs:** the perft gate includes the 960 suite.
- **Silent search bugs** (mate scores, draw scores under `d`, TT staleness): mate-in-N tests, `bench` determinism, asserts.

## B.13 Reading list

- **Board (CPW):** *Board Representation*, *Bitboards*, *Square Mapping Considerations*, *Magic Bitboards*, *BMI2*, *Move Generation*, *Zobrist Hashing*, *Repetitions*, *Fifty-move Rule*, *Perft Results*, *Chess960*, *Chess960 Perft Results*.
- **Search (CPW):** *Getting Started*, *Search Progression*, *Principal Variation Search*, *Transposition Table*, *Move Ordering*, *Quiescence Search*, *Static Exchange Evaluation*, *Null Move Pruning*, *Late Move Reductions*, *Futility Pruning*, *History Heuristic*, *Internal Iterative Reductions*, *Singular Extensions*, *Static Evaluation Correction History*, *Time Management*, *Contempt Factor*, *Fortress*, *Syzygy Bases*.
- **Eval and training (CPW):** *Tapered Eval*, *PeSTO's Evaluation Function*, *Texel's Tuning Method*, *NNUE*, *Stockfish NNUE*.
- **Testing (CPW):** *Engine Testing*, *Sequential Probability Ratio Test*.
- **Papers:**
  - Guid & Bratko, *Detecting Fortresses in Chess* (2012).
  - Fedus et al., *Switch Transformers* (2021).
  - Komatsuzaki et al., *Sparse Upcycling* (2022).
  - Shazeer et al., *Sparsely-Gated MoE* (2017).
- **Tools and data:**
  - fastchess (`man.md` for the `-engine`, `-sprt`, `-tb` options); Stockfish README (`UCI_LimitStrength`, `UCI_Elo`, `UCI_ShowWDL`); Ordo.
  - bullet; nnue-pytorch.
  - Fathom / Pyrrhic; Lichess tablebase API.
  - Lc0 training data and the lczero.org *Training data format versions* page.
  - Lichess database (CC0); the Lc0 v0.30 WDL contempt write-up.
  - Kaggle CLI (`kaggle kernels push/output`, `kaggle datasets version`).

---

## Decision log

| Decision | Why |
|---|---|
| Testing infrastructure (UCI, `bench`, SPRT) before any search feature | Every feature is SPRT-gated, so the gate must exist first. |
| Defensive specialty with its own benchmark | It is the chosen focus, and the benchmark makes it measurable. Defense targets known engine weaknesses (fortresses, the 50-move horizon, stalemate blind spots). |
| W/D/L output with a draw-value utility | A centipawn number can't tell a drawn fortress from a lost position. |
| Big capacity through routing (king buckets, MoE), not width | Width slows every move; routing adds parameters at constant compute. |
| PVS alpha-beta; no MCTS or MTD(f) | Fits a CPU engine with a value-only NNUE; novelty goes into eval and defense. |
| Chess960 from day 1 | Retrofitting castling is painful; the 960 perft suite strengthens the gate. |
| Lc0 + Stockfish + result blend (0.6/0.25/0.15) instead of own self-play data | Lc0's public data already has searched draw probabilities. This removes most of the CPU cost and speeds up the project. |
| Kaggle CPU sessions for data, laptop for timed tests, server optional | Server permission is uncertain; Kaggle CPU is free (5 parallel sessions); Colab is GPU-only. |
| Stockfish as the opponent: node-limited for the benchmark, `UCI_Elo` at 60+0.6 for rating | Node limits give accurate, reproducible pressure. `UCI_Elo` is calibrated to CCRL at exactly that time control. |
| Checkpoint/resume spec | 12-hour sessions and weekly quotas make resumable training mandatory. |

## Continue in a new chat

Paste this whole file as the first message, then try:
1. "Give me the Phase 0-1 checklist: repo layout, CMake targets, Position struct with Chess960 castling, magic finder, perft + 960 harness."
2. "Write the fastchess scripts for every row of the B.6 time-control table, plus the bench positions and a results log format."
3. "Write the search()/qsearch() skeleton with the search stack, TT, mate scores, and hooks for the draw-value utility and tablebase probes."
4. "Write the Kaggle data-pipeline notebook (Lc0 v6 converter + filters + Stockfish relabel at 2k nodes + manifest), the 40-byte record format, and train.py with the checkpoint/resume spec."
5. "Design the WDL head loss, the N2 king-bucket layout, and the N3 router + load-balancing loss in PyTorch, plus the C++ inference."

**Open items:** none blocking. The server stays optional; ask whoever manages it before using it.
