# Project B: Defensive NNUE Chess Engine (plan v3)

## Context for the new chat

- **Who:** Final-year/fresher candidate targeting SDE and AI/ML roles. Strong in C++ (research internship: RDKit batch featurization with OpenMP and Boost.Python), Go (distributed social feed engine), applied ML (RNA classification, agentic chemistry toolchain), and AWS (Amazon HackOn project).
- **Goal:** Add breadth to the resume with advanced, measurable projects. Resume advice already given: keep 4 projects, order by strength, add a skills section, fix LaTeX issues (stray `\end{contained}`, empty `\href{}`), and drop the chess project already commented out in favor of the new engine.
- **Decisions made:**
  1. Project A: a transactional sandbox for coding agents, extended into an **agent eval harness**. Kept as the headline project.
  2. Project B: a **bitboard C++ chess engine with a trained NNUE evaluation**, now with a clear identity: **a defensive specialist**. It recognizes fortresses, steers lost positions toward draws (repetition, stalemate, 50-move rule, tablebase draws), and proves it with a defense benchmark.
  3. The net is **big in capacity but sparse in compute**: king-bucketed input layer, WDL (win/draw/loss) output, and a **mixture-of-experts** experiment after the accumulator.
  4. **Chess960 / DFRC supported** from day 1.
  5. **Training labels come from teachers plus own data.** The main nets are distilled from **Leela Chess Zero's public self-play data** (ODbL licence; per-position MCTS win/draw/loss), blended with **Stockfish** search scores computed on the server and with game results. AlphaZero itself was never released; Lc0 is its open reproduction. A from-scratch self-play track remains as an experiment.
  6. Dropped for now: mini LLM inference engine, Raft KV store, LoRA fine-tuning project.
- **Hardware:**
  - **Server (the main machine):** Intel Xeon Silver 4314 (Ice Lake-SP), 16 cores / 32 threads, 2.4 GHz base / 3.4 GHz max turbo, 251 GB RAM. It supports AVX-512 with VNNI, and BMI2 with fast PEXT. It runs SPRT tests, data generation, Stockfish relabeling, Lc0 data conversion and shuffling, and CPU training of small nets.
  - **Laptop:** 8 CPUs, WSL2, Linux 5.15. Used for development and quick tests.
  - **Kaggle:** GPUs (P100 or 2x T4, about 30 GPU-hours/week, 12-hour sessions) for training. CPU sessions (at most 5 concurrent batch sessions, 12 hours each) for extra data generation.
  - **Colab free tier:** GPU fallback for training only.
  - Training must be **checkpointed and resumable across server, Kaggle and Colab** (B.9).
- **Sequencing:** about 25 weeks in total for A and B at 10-15 hours per week, overlapped. The engine is resume-complete after **Phase 3** (the fallback). The expanded scope here is about 20 weeks of B work, so B.10 marks what to cut.
- **Dev environment:** keep source, binaries and datasets on the Linux filesystem (`~/...`), not `/mnt/c`.
- **This file covers Project B only.** Older versions: `project_B_nnue_chess_engine.v1.md` (original) and `.v2.md` (research-based restructure).

### What changed from v2

| Change | Why |
|---|---|
| The engine's identity is **defense**, with its own benchmark (B.1, B.6) and a defensive search module (B.4) | A crowded project space needs a measurable specialty. Defense plays to known engine weaknesses: fortress blindness, the 50-move horizon, and stalemate blind spots in qsearch. |
| The net outputs **W/D/L**, not one scalar; search optimizes `W + d*D` | A centipawn number cannot tell "-3 but a dead-draw fortress" from "-3 and lost". Draw probability is the quantity a defender needs. |
| "Big net" means **big capacity, constant compute**: king input buckets (hard-routed experts) plus post-accumulator MoE | Widening the first layer slows every move linearly. Routing adds parameters without adding per-eval work. |
| **Learned-router MoE** is an experiment against Stockfish-style piece-count layer stacks, at equal active compute | It is a genuine ML research question with a fair baseline. B.8 explains where experts can and cannot go in an NNUE. |
| A **search-technique taxonomy** with recommendations (B.4) | It answers "which search?" once: PVS alpha-beta, plus why not MCTS or MTD(f). |
| Chess960 / DFRC from day 1 | Datagen diversity, plus the 850-position CPW Chess960 perft suite. Retrofitting castling later is painful. |
| A Kaggle/Colab **checkpoint and resume spec** (B.9) | 12-hour sessions, weekly quotas and Colab disconnects make resumable training mandatory. |
| **Syzygy tablebases** (3-5 piece, about 939 MiB) via Fathom, used in search, in datagen and for label rescoring | Exact draw knowledge in endgames is the cheapest "defensive monster" win. |

### Update v3.1: server + teacher data

| Change | Why |
|---|---|
| The **server** handles SPRT, data generation, relabeling, conversion and small-net training; core pinning keeps match timing stable (B.9) | About 4-6x the laptop's data-generation throughput, it runs 24/7, and its 251 GB of RAM holds 1B+ training records for global shuffling. |
| Main labels: **Lc0 MCTS W/D/L + Stockfish score + game result**, blended as soft targets (B.7) | Lc0's public data already contains a searched draw probability for billions of positions, which is exactly what a defense-focused WDL net needs, at zero labeling cost. |
| Data generation is **not on GPU** (B.9) | Alpha-beta search with incremental NNUE updates is branchy and evaluates one position at a time. GPUs need large batches (MCTS, bulk relabeling). GPU hours go to training. |
| Kaggle CPU sessions are a **supplement** (5 concurrent batch sessions), not unlimited compute; **Colab is not used for data generation** | Kaggle caps concurrent sessions. Colab's FAQ prioritizes interactive use and forbids using multiple accounts to get around limits. |
| A separate **AVX-512/VNNI build** for the server | Faster first-layer updates and int8 layers mean faster data generation and higher test throughput. |

---

## B.1 Goal and success criteria

A UCI-compliant (standard + Chess960), bitboard C++ engine with a tuned PVS search and a self-trained WDL NNUE. It is specialized for defense, every change is SPRT-validated, and results are reported against calibrated anchors.

**What "defensive monster" means, measurably** (the defense benchmark, B.6):
1. **Save rate:** from unbalanced openings (UHO book, roughly +1 for one side), the engine plays only the worse side against a fixed reference opponent. Report draws + wins, compared with a non-defensive baseline build of the same engine at equal overall Elo, with 95% CI.
2. **Fortress suite:** the share of curated fortress positions where the engine (a) evaluates near a draw (P(draw) high, |eval| small) within a fixed time, and (b) holds the draw in play against a strong attacker.
3. **Tablebase defense:** 6-7 piece positions that are verified draws despite a material deficit, played with only 5-piece tablebases loaded: the hold rate. In lost positions: plies survived compared with DTZ-optimal resistance.
4. **No collapse in overall strength:** non-regression SPRT against the non-defensive build on a balanced book. Specialization must not cost more than a few Elo, or the tradeoff is reported explicitly.
5. **WDL calibration:** predicted draw probability vs. actual draw frequency on held-out games (reliability diagram, expected calibration error).

| Milestone | Measurable outcome | Rough Elo prior (CCRL Blitz scale; replace it with your anchor-gauntlet number) |
|---|---|---|
| Phase 1: movegen | Perft exact on the 6 CPW positions + the EPD suite + the Chess960 suite; perft Mnps | n/a |
| Phase 3: HCE engine + tablebases | Elo-per-feature SPRT table, anchor gauntlet | ~2200-2600 |
| Phase 5: first NNUE | +Y Elo over HCE (fixed-time **and** fixed-node) | ~2700-3100 |
| Phase 6: defense | Save rate +X points over baseline at no Elo loss; fortress and tablebase hold rates; calibration | n/a (style metric) |
| Phase 7: MoE | Learned router vs piece-count stacks vs dense, at equal active compute | Report Δ, including negative results |

On stalemate tricks: search already scores stalemate as a draw. Against **engines**, what saves games is accurate draw prediction plus a search that steers toward draws, not "traps". Swindles that rely on an opponent blundering only pay off against **humans**, so they are a stretch goal for a Lichess bot mode (B.4, D8).

## B.2 Mental model: three layers and where they couple

1. **Board representation and move generation:** correctness first, speed second. A bug here silently poisons search, testing and training data.
2. **Search:** makes the static eval more accurate by looking ahead. Modern strength comes from *selectivity* driven by good move ordering. **Style** lives here, as a utility function plus draw handling.
3. **Evaluation:** `eval(position)` returns W/D/L probabilities. Search turns them into one scalar using the engine's draw value.

Where the layers couple:
- **Eval scale ↔ pruning margins:** all margins are in centipawns, so changing the eval or the draw value means retuning them.
- **Eval speed ↔ depth:** report fixed-node and fixed-time results separately.
- **Movegen correctness ↔ data quality:** this is what the perft gate protects.
- **TT ↔ path-dependence:** mate-score ply adjustment, repetitions, the 50-move counter, and now the draw value `d`. TT scores depend on `d`, so bump the TT age (or clear) when `d` changes between moves.
- **Data neutrality:** datagen runs with a **neutral** draw value (`d = 0.5`). Style belongs in play-time search, not in training labels, or the net learns a biased draw probability.

Win probability: `sigmoid(cp/400)`. A cp value maps back as `cp = 400*logit(p)`.

---

## B.3 Board representation (decided)

Modern engines are **hybrid**: bitboards for move generation, attacks and eval, plus a mailbox for O(1) "what's on square x".

| Decision | Choice | Notes |
|---|---|---|
| Square mapping | **LERF**: a1 = 0 ... h8 = 63, north = `<< 8` | `sq ^ 56` flips ranks (black's perspective in PSTs and NNUE). |
| Piece storage | 6 piece-type bitboards + 2 color bitboards + `Piece mailbox[64]` | One AND per "white knights" query. |
| Sliding attacks | **Fancy magic bitboards**: about 800 KiB rooks + 38 KiB bishops. Find your own magics (random sparse candidates, collision-checked). | Optional PEXT build flag: fast on Intel Haswell+ and Zen 3+, very slow on Zen 1/2. |
| Other tables | Knight, king and pawn attacks; `between[64][64]`, `line[64][64]` | Pins, check blocks and castling paths each become one AND. |
| Move encoding | 16-bit: from (6), to (6), flags (4) | **Castling is encoded as king-captures-own-rook** (needed for Chess960). |
| Make/unmake | **Copy-make** into a ply-indexed stack | NNUE accumulators are per-ply anyway. |
| Legality | Pseudo-legal staged generation + `is_legal(move)` with precomputed `checkers` and `pinned` bitboards; a separate evasion generator | Fits staged ordering. |
| Zobrist | 12x64 + side + 16 castling-mask keys + 8 en passant file keys; **hash the en passant file only if a capture is possible**; pawn and non-pawn sub-keys | Debug assert: incremental key == recomputed key. |
| Draws | Repetition (scan back to the last irreversible move, step 2), 50-move rule (mate on move 100 takes precedence), insufficient material | Single repetition = draw inside the tree; threefold against game history. |

### Chess960 / DFRC (supported from day 1)
- Store `castling_rook_square[4]`. The castling-rights mask table covers the king square and each castling-rook square.
- **Final squares are fixed:** king to g1/c1 (g8/c8), rook to f1/d1 (f8/d8). The king may not move at all, and the rook may already be on its destination.
- **Legality:** every square between the king and its destination, and between the rook and its destination, is empty except for that king and that rook. The king is not in check, and every square the king passes through, destination included, is unattacked. **Compute those attacks with the castling rook removed from occupancy.** Example: rook b1 shields the king from a queen on a1; after queenside castling the king on c1 would be in check.
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
| MTD(f) | Plaat's MTD; some older engines | Chains of null-window searches that lean hard on the TT. Interacts badly with LMR, singular extensions and aspiration-style tuning; no modern top engine uses it. | No |
| Best-first: SSS*, B*, conspiracy numbers | Historical / research | Memory-heavy, superseded | No |
| Proof-number search | Mate and endgame solvers | Proves binary outcomes; impractical for fortresses with the 50-move rule | Not in the engine (at most an offline mate-suite tool) |
| MCTS / PUCT | AlphaZero, Lc0 | Needs a **policy** head and GPU-sized nets; weak tactically at CPU speed with a value-only eval. It is WDL-native, but the WDL idea can be borrowed without the algorithm. | No (borrow WDL only) |
| MCTS/alpha-beta hybrids | Komodo MCTS mode, Scorpio | Large complexity for uncertain gain | No |
| Parallel: Lazy SMP vs YBWC / ABDADA | Stockfish switched to Lazy SMP in 2016 | Lazy SMP is simple and scales well to about 8+ cores | Stretch only |

**Recommendation:** PVS alpha-beta with the full selectivity ladder below, plus a defensive module (D1-D8) on top. Spend novelty budget on evaluation, draw-awareness and the MoE experiment, not on an exotic search algorithm.

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
| 14 | Correction history | Pawn, material and non-pawn keyed EMA of `search - static`. Doubles as a "position complexity" signal. |
| 15 | Qsearch futility / delta | Skip captures that cannot reach alpha. |

Then SPSA-tune the constants. Retune again after the NNUE switch and after enabling the draw-value utility.

### Defensive module (the specialty)
Measured on the defense benchmark (B.6) **and** a non-regression SPRT for overall strength.

| # | Technique | What it does | Notes |
|---|---|---|---|
| D1 | **WDL eval → utility** | The net gives W, D, L for the side to move. The engine optimizes `U = W + d*D` from its own perspective. Search uses `cp = 400*logit(U)`. At opponent-to-move nodes the draw weight is `1 - d`, which keeps negamax consistent (`cp_opp = -cp_eng`). | `d = 0.5` is neutral (expected score). |
| D2 | **Situational draw value (contempt)** | At the start of each move, set `d` from the previous root WDL: neutral by default; `d` around 0.6-0.7 when root P(loss) is high ("defense mode"), or against a stronger opponent (UCI options `DrawValue`, `DefenseMode`). In-tree draws then score `400*logit(d)` for the engine. | Never use a large `d` globally: `d = 0.6` values a draw like +162 cp, so the engine would chase draws from equal positions. |
| D3 | **50-move awareness** | Scale the eval toward the draw score as the halfmove clock grows, Stockfish-style (`v * (212 - rule50) / 212`). Shuffling then reads as "holding". | Ablation: post-hoc scaling vs. feeding `rule50` to the net (B.8). |
| D4 | **Syzygy tablebases** (Fathom, MIT, or its fork Pyrrhic) | 3-5 piece tables (about 939 MiB) on disk. Probe WDL in search (not qsearch) when pieces <= 5, `rule50 == 0` and no castling rights. At the root, keep only drawing or winning moves by DTZ. When lost, play the **max-DTZ** move (longest resistance) and treat **blessed losses** (lost, but saved by the 50-move rule) as draws. | 6-piece tables are 149 GiB, optional. |
| D5 | **Stalemate guard in qsearch** | Stand-pat assumes a quiet move exists. For low-material positions where the side to move has a king plus few or blocked pieces, check whether any legal move exists; if none, score it a draw. Fixes a classic blind spot. | Cheap; gate by a material threshold. |
| D6 | **Perpetual-check seeking** | When the side to move is the defender and far below alpha, search quiet checks at the first qsearch ply and extend checks, to find perpetuals beyond the horizon. | SPRT carefully, because check searching inflates trees. |
| D7 | **Fortress / no-progress detection** (Guid & Bratko 2012) | At the root: if \|score\| is large but flat across many iterations and the PV contains no zeroing move (pawn move or capture), the stronger side is making no progress, so scale toward a draw. | Experimental; judged on the fortress suite. |
| D8 | **Swindle mode (humans only, stretch)** | Among moves with nearly equal `U`, prefer ones where many opponent replies lead to draws (stalemate traps, perpetuals). | Only for a Lichess bot; worthless against engines. |

### Time management
`soft = remaining/20 + inc/2`; hard limit a few times soft, capped well below remaining, minus `Move Overhead`. Check soft between iterations and hard every N nodes. Upgrades: node-based scaling, best-move stability, eval stability. **Defense extra:** spend more time when root P(loss) jumps, since that is where saves are won.

### Parallel search
Lazy SMP is a stretch goal. Testing and datagen run many single-threaded games in parallel instead.

---

## B.5 Handcrafted evaluation: small on purpose

Its jobs: the Elo baseline, and the gen-0 data generator. Timebox it.
- Tapered PeSTO material + PSTs, phase N = B = 1, R = 2, Q = 4 (total 24): `(mg*phase + eg*(24 - phase)) / 24`.
- Optional terms, each SPRT'd: mobility, passed pawns, pawn structure, bishop pair, king safety. Plus **drawishness scaling** (opposite-colored bishops, rook-pawn + wrong bishop, low-material endings), which is a cheap first taste of the defense identity.
- **Texel tuning as the trainer's first model:** `E = mean((R - sigmoid(K*q/400))^2)` in PyTorch, on the shared data format.
- **Symmetry test:** `eval(pos) == eval(color_flip(pos))`.

---

## B.6 Testing and measurement

- **Git from day 1.** Every functional commit carries `Bench: <nodes>`. The `bench` command runs a fixed-depth search over about 50 FENs, standard and 960.
- **Correctness suite:** perft (standard + 960), FEN round-trip (X-FEN and Shredder-FEN), Zobrist check, SEE tests, eval symmetry, mate-in-N, tablebase probe sanity, and a float-vs-quantized net check. Debug builds use asserts and `-fsanitize=address,undefined`.
- **Match runner:** fastchess (pentanomial SPRT), single-threaded engines, Hash 16-32 MB.
  - Runs on the **server**, pinned to 8 physical cores with their hyperthread siblings left idle (`taskset`; read the sibling map from `lscpu -e`), giving 8 concurrent games. Use 15 concurrent games when nothing else is running.
  - Data generation and relabeling use the other 8 cores x 2 threads. **Never share a physical core between a match and data generation**, because contention causes time losses and noise.
  - Both builds in a test use the same compiler flags on the same machine.
  - The laptop runs quick sanity matches only.
- **Time controls:** STC 8+0.08; LTC 40+0.4 for search and defense changes, because defensive gains often grow with depth.
- **Books:** `8moves_v3.pgn` while weak, then `UHO_Lichess_4852_v1.epd`. A **DFRC book** for 960 tests (`-variant fischerandom`).
- **SPRT bounds** (alpha = beta = 0.05, LLR ±2.94): weak engine `[0, 10]` / `[-10, 0]`; later `[0, 5]` / `[-5, 0]`.
- **Anchor gauntlet** at the end of Phases 3, 5, 6 and 7: 1000+ games against 3-4 engines with known CCRL ratings.

### Defense benchmark (built in Phase 6, run on every defense change)
| Test | Setup | Metric |
|---|---|---|
| Save rate | UHO positions; engine plays **only the worse side** against a fixed reference opponent (e.g. a stronger engine or a longer-TC build). Run baseline and candidate on the same positions. | (draws + wins) / games, with a 95% CI on the difference |
| Fortress suite | Curated from the CPW *Fortress* examples, Guid & Bratko (2012), and self-play positions that were drawn despite \|eval\| > 200 | % recognized (P(draw) > 0.6 or \|eval\| < 50 cp within T s); % held in play over 100 moves |
| Tablebase defense | 6-7 piece positions that are drawn despite a material deficit, verified with the **Lichess tablebase API** (`tablebase.lichess.ovh/standard?fen=...`, which covers 7 pieces; cache and throttle requests). Engine has only 5-piece TBs. | Hold rate against a strong attacker; for lost positions, plies survived vs DTZ |
| Calibration | Held-out self-play games | Reliability diagram, expected calibration error for P(draw) |
| Strength guard | Balanced book, SPRT `[-5, 0]` against the non-defensive build | Pass/fail plus Elo ± error |

No standard public fortress suite is assumed. Building and publishing one is itself a contribution.

---

## B.7 Labels: multi-teacher WDL distillation

### Label sources per position
| Source | What it provides | Cost | License / disclosure |
|---|---|---|---|
| Game result | W/D/L one-hot. Noisy, but grounded in what actually happened. | Free | n/a |
| **Lc0 public training data** (v6 chunks from `storage.lczero.org/files/training_data/`) | `root_q`, `root_d`: MCTS value and draw probability at the root after search. Also `best_q/best_d`, the game result (`result_q/result_d`), `rule50`, and the board as 104 bit-planes. | Download + convert; billions of positions exist | **ODbL**: credit the source; any derived dataset you publish must also be ODbL. Disclose Lc0 as a teacher. |
| **Stockfish relabel** (run on the server) | Score in centipawns at fixed nodes, giving expected score and, via Stockfish's WDL model (`UCI_ShowWDL`), W/D/L | Roughly 50-100M positions/day on half the server at ~10k nodes (estimate; measure) | The output generally isn't GPL-covered (its code and nets would be). Disclose it as a teacher. |
| Syzygy (<= 5 pieces) | Exact W/D/L, aware of the 50-move rule | About free | n/a |
| Own engine | Score in centipawns at ~5k nodes, on positions *your* search visits | Data-generation compute | Yours |

**AlphaZero itself is not available.** DeepMind never released weights or data. **Lc0 is the open reproduction**, and its data is better for this project than running a teacher yourself, because a searched draw probability is already stored for every position.

### Turning labels into targets
- **Lc0 q/d → W/D/L:** `W = (1 + q - d)/2`, `D = d`, `L = (1 - q - d)/2`, from the side to move's view. **Unit-test the perspective** on known won, lost and drawn positions before training.
- **Blend as soft labels:** `t = a*WDL_lc0 + b*WDL_sf + c*onehot(result)`, with `a + b + c = 1`. Exact tablebase W/D/L overrides everything at <= 5 pieces, with cursed win and blessed loss both counting as a draw.
- **WDL net loss:** soft cross-entropy `CE(softmax(z), t)` (knowledge distillation) plus an optional `alpha*((W + D/2) - E_target)^2` expected-score term.
- **Scalar N1:** `target = lambda*E_teacher + (1 - lambda)*result`, where `E_teacher = W + D/2` of the blend.
- **Why blend:**
  - Lc0's MCTS is strong strategically and models draws, which is what a defender needs.
  - Stockfish's alpha-beta is tactically sharp at small budgets.
  - The game result anchors both to reality.
  - The best mix is an **experiment** (B.11), not an assumption. Starting point: `a = 0.6, b = 0.25, c = 0.15`.

### Gotchas
- **Distribution shift:** Lc0 positions come from Lc0 self-play, not from the positions your search visits. Also have Stockfish relabel some positions from your own engine's games (ablation in B.11).
- **Filtering:**
  - Skip early plies (the opening-book region) and in-check positions.
  - Skip positions whose best move is a capture or promotion. Decode `best_idx` with Lc0's 1858-move policy table.
  - Skip extreme `|q|`.
- **Size:** a raw v6 record is 8,356 bytes, mostly the policy array. Stream it: download a tar, convert to about 40-byte records (B.9), delete the tar.
- **Ready-made shortcut:** Stockfish's training binpacks (Leela-derived; links at `robotmoon.com/nnue-training-data`) are compact and huge. But they store only centipawns + result (no draw probability) and need a binpack reader. nnue-pytorch's reader is GPL-3; if you reuse it, release the trainer under GPL-3.

### Data tracks (always reported separately)
- **Teacher-distilled:** the main nets; teachers disclosed in the README. Stockfish itself trains on Leela data, so this is mainstream, though some testers and events distinguish engines trained on original data.
- **Own self-play bootstrap:** the from-scratch experiment. Gen-0 comes from the HCE, then each net generates data for the next, giving an Elo-per-generation curve.
- **Data generation stays neutral (`d = 0.5`)** in every track. The defensive style lives only in play-time search.

---

## B.8 NNUE design: big capacity, sparse compute

### Cost model (why "big" must mean sparse)
For L1 width `N`:
- Accumulator update per move: about (2-4 changed features) x N x 2 perspectives int16 adds.
- Post-accumulator layers: about 2N x 16 int8 MACs.
- Doubling N roughly doubles both, so the engine searches shallower at fixed time.
- **Routing (bucket or expert selection) multiplies parameters without multiplying per-eval work.**

### Architecture roadmap
| Net | Input layer | After the accumulator | Output | Phase |
|---|---|---|---|---|
| **N1** | `768 -> 256` (x2 perspectives), SCReLU | none | 1 scalar | 4-5 (pipeline proof; matches bullet's `simple` example for cross-checking) |
| **N2 "big"** | **Horizontally mirrored, 8-16 king input buckets**, L1 512-1024. Finny tables for bucket-change refreshes. | `2N -> 16 -> 32` | **3 (W/D/L)** | 6 |
| **N3 "MoE"** | Same as N2 | **K = 8 expert stacks** `2N -> 16 -> 32 -> 3` + router | 3 | 7 |

Rough sizes: N2 with 16 buckets x 768 x 1024 is about 12.6M first-layer parameters (about 25 MB int16), while per-eval compute is the same as an unbucketed 1024-wide net. **King input buckets are already a hard-routed sparse MoE at the first layer.**

### Where experts can and cannot go
- **First layer, learned routing: no.** Every expert's accumulator would need incremental updates on every move, so K experts cost Kx the update work and the sparsity is lost. The same objection applies to "K full NNUEs + a gate" designs; one TalkChess thread claims such a result, but I could not verify it.
- **First layer, hard routing by king square: yes.** These are the N2 input buckets.
- **After the accumulator, learned routing: yes.** The SCReLU accumulator output is already computed, so a router costs about `2N x K` int8 MACs, comparable to the `2N -> 16` layer, and only one expert stack runs.

### N3 experiment design
- **Baselines at equal active compute:** (A) dense single stack; (B) **8 hard stacks by piece count** (Stockfish-style layer stacks); (C) **learned top-1 router over 8 stacks**; optional (D) learned top-2 (2x stack cost).
- **Router input:** accumulator output plus a few cheap scalars the accumulator lacks: phase, material imbalance, `rule50/100`, side to move. This lets the router route on drawishness signals.
- **Training (Switch Transformer recipe):** top-1, output `y = p_i(x) * E_i(x)` both in training and in the engine (a float softmax over 8 logits is negligible). **Load-balancing loss** `alpha * K * sum_i(f_i * P_i)`, alpha around 0.01. Optional router z-loss.
- **Sparse upcycling:** initialize all 8 experts from the trained N2 dense stack, then train the router and experts. This needs far less data than training from scratch.
- **Quantization:** router weights int8 on uint8 activations, same as the `2N -> 16` layer; argmax picks the expert.
- **Risks to measure:**
  - **Eval discontinuity** at routing flips: flip rate along game trajectories and the |Δeval| distribution. It can destabilize aspiration windows and the TT.
  - **Expert collapse:** routing utilization and entropy.
  - **Data per expert:** about 1/K of the data each.
- **Metrics:** validation loss, WDL calibration, fixed-node Elo, fixed-time Elo, NPS, defense benchmark, and an **expert specialization analysis** (phase, material, imbalance and draw rate per expert). Does an emergent "defense expert" appear? A clean negative result ("learned routing does not beat piece-count stacks at this data scale") is still a strong write-up.

### Quantization and inference (all nets)
- First layer: QA = 255, int16, weights clamped to ±1.98 in training (the SCReLU `v*w` int16 trick).
- Hidden layers: int8 (QB = 64 for N1's output).
- Lazy accumulator updates on a per-ply stack, fused add/sub, AVX2 with a scalar fallback.
- WDL head: 3 logits, then a float softmax.
- Always verify: C++ float matches PyTorch, then quantized matches float, on 10k positions, before measuring Elo.

### Data scale
More parameters need more data: N1 needs about 50-100M positions; N2 and N3 want **1B+**. Bucketed and MoE nets spread data thinner, which is another reason to use upcycling. With Lc0 data, a billion positions is a download-and-convert job (B.9), so the scarce resource becomes **GPU training hours**, not data.

---

## B.9 Compute plan, data pipeline and training

### Which machine does what
| Workload | Where | Notes |
|---|---|---|
| Development, unit tests, quick perft | Laptop | |
| SPRT + defense-benchmark matches | **Server**, 8 physical cores pinned, hyperthread siblings idle | Stable timing (B.6) |
| Lc0 download + conversion, Stockfish relabeling, own data generation, global shuffle/dedupe/split-by-game | **Server**, the other 8 cores x 2 threads, plus RAM | Run with `nice` if the server is shared |
| Extra own-engine data generation | **Kaggle CPU batch sessions**: at most 5 at once, 12 h each, a few vCPUs each. Launch with `kaggle kernels push` and distinct seeds; collect with `kaggle kernels output`. | Real but capped. One account only. Roughly adds half a server's worth while kept relaunched (estimate). |
| N1 training and small sweeps | Server CPU (PyTorch, data held in RAM) or Kaggle GPU | Unlimited hours on the server, just slower |
| N2/N3 training and sweeps | Kaggle GPU (P100 / T4); Colab fallback | 30 GPU-h/week is now the scarcest resource. Spend it on training. |
| Colab | GPU fallback for training only | Its FAQ says Colab "prioritizes interactive compute" and forbids using multiple accounts to get around limits. Unattended data generation is the wrong use. |

**Why data generation doesn't run on a GPU:** alpha-beta search with incremental NNUE updates evaluates one position at a time, deep inside a branchy recursive search. GPUs only pay off with thousands of independent evaluations batched together. They suit MCTS engines (that is how Lc0's data was made) and bulk relabeling with a large net. Optional extra: label your own engine's positions with an Lc0 net's W/D/L using lc0's CUDA backend, but only if GPU hours are left over after training.

### Engine builds
- **Portable build:** `-march=x86-64-v3` (AVX2 + BMI2).
- **Server build:** `-march=icelake-server`. It uses AVX-512BW for int16 first-layer updates, **VNNI (`vpdpbusd`)** for the int8 layers, and fast PEXT for sliding-piece attacks (fast on Intel).
- Data generation uses the server build. Report NPS together with the CPU model.

### Throughput (estimates; measure in Phase 4)
- **Own-engine data generation at 5k nodes per move:**
  - All 32 server threads: roughly 150-250M positions/day. All-core turbo sits below the 3.4 GHz maximum, and hyperthreading adds roughly 20-30% for chess search.
  - Half the server (the normal split): about 75-125M/day.
  - The laptop by comparison: about 40M per night.
- **Stockfish relabeling** at about 10k nodes: roughly half the rate per thread.
- **Lc0 data** is limited by download bandwidth and disk, not CPU. Stream it: download a tar, convert it, delete it.
- **RAM:** 1B records x 40 B = 40 GB, which fits in the server's 251 GB. Do the global shuffle and the split-by-game validation set in memory.

### Own-engine data generation (from-scratch track and on-distribution positions)
- **Openings mix** (itself an ablation): standard start + 8-9 random plies; **Chess960/DFRC starts** + random plies (diversity); **UHO-style unbalanced positions** (lots of defense practice for the losing side). Discard lopsided starts (|eval| > ~1000 cp).
- Play with a soft limit of about 5k nodes/move at **neutral `d = 0.5`**. Adjudicate wins (|score| >= ~2000 for several plies) and draws. **End games at <= 5 pieces with the tablebase result.**
- Filter at load time: in check, capture or promotion best move, extreme scores. **Up-weight** "drawn but imbalanced" positions as an ablation.

### Record format (multi-label, about 40 bytes, fixed-size, versioned shard header)
- **Board:** occupancy bitboard (8 B) + 4-bit piece codes (16 B) + side to move / castling / en passant / `rule50` (3 B).
- **Labels:**
  - Own-engine score (int16 cp)
  - Stockfish score (int16 cp; sentinel if absent)
  - Lc0 `q` (int16, scaled) and `d` (uint16, scaled)
  - Result W/D/L + tablebase flag + source id (1-2 B)
- **Reserved** bytes for later fields.
- Shards of about 1 GB (about 25M records). A `to_bulletformat` converter feeds bullet for the N1 cross-check.

### Where everything lives
| Item | Store | Why |
|---|---|---|
| Master dataset | **Server disk** (all records) | Shuffling, dedupe, mixing |
| Training subsets | **Private Kaggle Dataset** `nnue-data`, one version per training mix (e.g. 300-500M records, 12-20 GB) via `kaggle datasets version -p shards/ -m "mix a0.6 b0.25 c0.15"` | Attaches read-only at `/kaggle/input/...`. Colab pulls the same data with the Kaggle API. Check the current per-dataset size limit. |
| Checkpoints | Kaggle: `/kaggle/working/ckpt` (persisted as notebook output when the run is a **committed "Save & Run All" version**), mirrored to a private Kaggle Dataset `nnue-checkpoints`. Colab: Google Drive. | The next Kaggle run adds the previous run's output as an input. Colab resumes from the checkpoints dataset. |
| Exported nets + lineage | Local repo `nets/` + `nets/registry.csv` | The engine side never depends on cloud state. |

Platform limits to design around:
- **Kaggle:** about 30 GPU-h/week (resets weekly), 12-hour max session, P100 (16 GB) or 2x T4, few CPU cores, `/kaggle/working` about 20 GB. API tokens go in **Kaggle Secrets**.
- **Colab free:** T4, up to about 12 h, idle disconnects, no guaranteed availability.
- Run the 12-hour jobs as **committed background runs**, not interactive sessions.

### Checkpoint and resume spec
```
ckpt/run-<id>/
  config.yaml          # arch, data generations + shard list, lambda/WDL weights, LR schedule, seed, git sha
  step-<n>.pt          # keep the last 3
  best.pt              # best validation loss
  metrics.jsonl        # step, train/val loss, calibration (ECE), LR, positions seen, router utilization
```
- **Checkpoint contents:** model, optimizer, LR scheduler and AMP scaler state; global step and superbatch; **data cursor** (shard-order seed, shard index, record offset); RNG states (python, numpy, torch, cuda); best validation loss; wall-clock used.
- **When:** every ~20 minutes and at every superbatch end. **Atomic write:** write a temp file, fsync, then `os.replace`.
- **Deadline-aware:** `--deadline 11h30m` makes a final checkpoint and exports a net before the platform kills the session. There is also a SIGTERM handler.
- **Resume:** `train.py --resume auto` searches `/kaggle/input/*/ckpt`, then Drive, then the server/local `ckpt/`, and refuses a config-hash mismatch unless `--allow-config-change` is passed (for intended LR or fine-tune changes).
- **Deterministic data order:** the shard permutation comes from the seed, so the cursor alone determines what comes next. Resume reproduces the same stream.
- **Export:** `export.py <ckpt>` writes `net-<sha8>.nnue` (quantized) plus `ref-<sha8>.jsonl` (float outputs on 10k verification FENs). The engine's `verifynet` command compares against it. Registry row: net id, parent checkpoint, data, architecture, val loss, ECE, fixed-node and fixed-time Elo, defense benchmark.
- **Loader:** a C++ (pybind11) decoder from packed records to feature indices, because Kaggle's few CPU cores will bottleneck any Python parsing. Pin memory and prefetch batches.
- **Reference trainer:** bullet (Rust; CUDA, ROCm or Metal) on Kaggle for N1 only, to cross-check your PyTorch trainer.
- **One trainer, three hosts:** the server (CPU, data in RAM), Kaggle and Colab all run the same `train.py` and checkpoint format, so a run can move between them mid-training. Rsync checkpoints between the server and the Kaggle checkpoints dataset.

---

## B.10 Phased plan (about 20 weeks of B time; cut lines marked)

| Phase | Weeks | Content | Exit |
|---|---|---|---|
| 0 | 1 (start) | Git, CMake (Release, Debug + sanitizers, portable), test runner | Builds and tests run |
| 1 | 1-3.5 | B.3 including Chess960: magics, staged pseudo-legal generator, copy-make, Zobrist, draws, FEN/X-FEN/Shredder-FEN | Perft gate (standard + 960 sample) |
| 2 | 4-5 | UCI (+ `UCI_Chess960`, Hash, Threads, Move Overhead), ID + PVS + qsearch + TT, PeSTO, basic time management, `bench`, fastchess SPRT pipeline | 1000 self-play games with no illegal moves or time losses; first SPRT passes |
| 3 | 5-8 | Selectivity ladder (at least steps 1-9), time-management upgrades, HCE + Texel, **Syzygy probing (D4)**, **rule50 scaling (D3)**, **qsearch stalemate guard (D5)** | **Fallback-complete:** Elo-per-feature table, anchor gauntlet, README |
| 4 | 9-10 | **Data pipeline:** Lc0 v6 converter (perspective unit tests), Stockfish relabeler, own-engine data generation (opening mix, tablebase adjudication), multi-label record format, shuffle/split on the server, Kaggle Dataset upload. **Trainer:** PyTorch + C++ loader + **checkpoint/resume**. N1 trained on the teacher blend, with float inference in C++. | N1 beats HCE at fixed nodes; a resume test (kill mid-run, resume, identical loss curve) passes |
| 5 | 11-12 | Quantized SIMD inference (AVX2 + AVX-512/VNNI), SPRT vs HCE, retune margins. Lc0 conversion and Stockfish relabeling **run continuously** on the server. | +Y Elo over HCE; gauntlet |
| 6 | 13-15 | **Defense:** N2 (king buckets + WDL head), D1-D2 utility, D6-D7, defense benchmark built and run, calibration | Save-rate gain with the strength guard passing |
| 7 | 16-18 | **MoE:** N3 variants A/B/C (+ D), upcycling, specialization analysis | Equal-compute comparison table |
| 8 | 19-20 | Own self-play bootstrap generations (from-scratch track; the server + Kaggle CPU sessions generate data in the background from Phase 5), final gauntlets, write-up (results page + blog-style README) | Resume line filled with real numbers |

**Cut order if behind:** (1) Phase 7 shrinks to A vs B only (hard stacks) or moves after Project A ships; (2) the own-bootstrap generations shrink to one generation; (3) D6-D8; (4) ladder steps 13-15; (5) Lazy SMP is never in scope. **Do not cut:** the perft gate, testing infrastructure, the WDL head, or the defense benchmark, because the specialty claim depends on them.

## B.11 Experiment menu

| Experiment | Holds fixed | What it shows |
|---|---|---|
| Save rate: neutral `d` vs situational `d` (D2) | Net, search | Value of the draw-preference knob |
| WDL head vs scalar net | Data, input layer | Whether explicit draw modeling improves defense and calibration |
| Post-hoc rule50 scaling vs rule50 as a net/router input | Net size | Learned vs hand-coded 50-move awareness |
| Tablebase rescoring of labels on/off | Data | Effect of exact endgame labels on fortress recognition |
| Opening mix (standard / DFRC / UHO share) | Data size | Diversity vs defense-practice tradeoff |
| Dense vs piece-count stacks vs learned router (vs top-2) | Active compute | The MoE question |
| Upcycled vs from-scratch MoE | Data | Data efficiency of upcycling |
| King-bucket count (1 / 4 / 8 / 16) | L1 width | Capacity via routing vs NPS (refresh cost) |
| Elo per bootstrap generation | Architecture | Self-improvement without a teacher |
| **Teacher blend grid** `(a, b, c)` for Lc0 / Stockfish / result | Positions, architecture | Which teacher matters for strength vs. for draw calibration and defense |
| Lc0-only vs Stockfish-only labels on the same positions | Positions | MCTS vs alpha-beta as a teacher |
| Lc0 positions vs own-engine positions (same labeler) | Labels, size | Effect of distribution shift |
| Teacher-distilled vs own self-play bootstrap | Size, architecture | Distillation vs learning from scratch |
| Own PyTorch trainer vs bullet (N1) | Data, architecture | Trainer correctness |

## B.12 Risks

- **Specialization costs Elo:** large `d` or aggressive draw-seeking throws away wins. Hence a situational `d`, the strength guard, and neutral-`d` datagen.
- **Scope creep:** about 20 weeks against a 25-week shared budget. Follow the cut order; the engine is complete at Phase 3.
- **MoE instability:** routing flips cause eval jumps and search instability. Measure them; fall back to piece-count stacks.
- **Label bugs in conversion:** a wrong perspective or sign in Lc0 `q`, or the wrong board orientation, silently ruins a net. Use unit tests on known positions, plus a sanity net trained on 10M positions before scaling up.
- **Licensing:** Lc0 data is ODbL (credit it; derived datasets you publish stay ODbL). Reusing nnue-pytorch code makes the trainer GPL-3. State the teachers in the README.
- **Distribution shift:** teacher positions are not the positions your search visits. Mix in relabeled own-engine positions.
- **Disk and bandwidth:** raw Lc0 tars are large. Stream-convert and delete. Check free disk on the server first.
- **GPU-hour scarcity:** 30 GPU-h/week is now the bottleneck. Run small sweeps on the server CPU and reserve GPUs for N2/N3.
- **Cloud fragility:** quotas, 12-hour kills, Colab disconnects. Deadline-aware checkpoints and deterministic resume; the training run is tested for resume in Phase 4.
- **CPU contention:** never share a physical core between matches and data generation. Pin with `taskset`. If the server is shared with other users, match timing gets noisy; check load before trusting an SPRT.
- **Movegen and 960 bugs poison data:** the perft gate includes the 960 suite.
- **Silent search bugs** (mate scores, draw scores under contempt, TT staleness when `d` changes): mate-in-N tests, `bench` determinism, asserts.
- **Self-play inflation:** anchor gauntlets.
- **Crowded space:** differentiated now by the defense identity + a measured benchmark + the MoE experiment + rigor.

## B.13 Resume target

"Defense-specialized chess engine in C++ (magic bitboards, Chess960, perft-verified on 950+ positions, X Mnps), PVS search with LMR, NMP, singular extensions, history heuristics and Syzygy probing; every change SPRT-tested. Distilled Lc0 (MCTS W/D/L) and Stockfish labels into a WDL NNUE (king-bucketed, ~12M params, int16/int8 AVX2/AVX-512-VNNI) on M positions with a fault-tolerant PyTorch pipeline across a 32-thread server and Kaggle/Colab GPUs, plus a from-scratch self-play track for comparison. Raised save rate from losing positions by X pts at equal Elo; calibrated draw prediction (ECE e); learned-router mixture-of-experts vs piece-count layer stacks at equal compute: Δ Elo. ~Z CCRL-equivalent."

## B.14 Reading list

- **Board (CPW):** *Board Representation*, *Bitboards*, *Square Mapping Considerations*, *Magic Bitboards*, *BMI2*, *Move Generation*, *Zobrist Hashing*, *Repetitions*, *Fifty-move Rule*, *Perft Results*, *Chess960*, *Chess960 Perft Results*.
- **Search (CPW):** *Getting Started*, *Search*, *Search Progression*, *Principal Variation Search*, *Transposition Table*, *Move Ordering*, *Quiescence Search*, *Static Exchange Evaluation*, *Null Move Pruning*, *Late Move Reductions*, *Futility Pruning*, *History Heuristic*, *Internal Iterative Reductions*, *Singular Extensions*, *Static Evaluation Correction History*, *Time Management*, *Lazy SMP*, *Contempt Factor*, *Fortress*, *Syzygy Bases*.
- **Eval and training (CPW):** *Tapered Eval*, *PeSTO's Evaluation Function*, *Texel's Tuning Method*, *NNUE*, *Stockfish NNUE*.
- **Testing (CPW):** *Engine Testing*, *Sequential Probability Ratio Test*.
- **Papers:**
  - Guid & Bratko, *Detecting Fortresses in Chess* (2012).
  - Fedus et al., *Switch Transformers* (2021): top-1 routing, load balancing.
  - Komatsuzaki et al., *Sparse Upcycling* (2022).
  - Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer* (2017).
- **Tools and data:**
  - bullet (github.com/jw1912/bullet); nnue-pytorch.
  - OpenBench; fastchess.
  - Fathom / Pyrrhic (tablebase probing); Lichess tablebase API (github.com/lichess-org/lila-tablebase).
  - Lichess database (CC0); the Lc0 v0.30 "WDL rescale/contempt" write-up (how a WDL engine turns a draw preference into play).
  - Lc0 training data (`storage.lczero.org/files/training_data/`, ODbL) and the *Training data format versions* page on lczero.org (v6 record layout).
  - Stockfish training-data index (robotmoon.com/nnue-training-data).
  - Kaggle CLI (`kaggle kernels push/output`, `kaggle datasets version`).

---

## Continue in a new chat

Paste this whole file as the first message, then try:
1. "Give me the Phase 0-1 checklist: repo layout, CMake targets, Position struct with Chess960 castling, magic finder, perft + 960 harness."
2. "Design the Phase 2 testing pipeline: bench positions (standard + 960), fastchess SPRT scripts, results log."
3. "Write the search()/qsearch() skeleton with the search stack, TT, mate scores, and hooks for the draw-value utility and tablebase probes."
4. "Design the datagen record format, the Kaggle Dataset layout, and train.py with the checkpoint/resume spec from B.9."
5. "Design the WDL head loss, the N2 king-bucket layout, and the N3 router + load-balancing loss in PyTorch, plus the C++ inference."

Open items:
- **Server disk and network:** free disk (the master dataset is about 40 GB per 1B records, plus scratch space for streaming Lc0 tars, plus 1 GB for the 5-piece Syzygy tables) and download bandwidth?
- **Server sharing:** is it yours alone, or shared? This affects whether 24/7 jobs are OK and whether match timing stays clean.
- **Laptop CPU model:** AVX2 support, for the portable build.
- **Reference opponent** for the save-rate benchmark: which engine and time control?
- **Teacher blend starting point:** `a = 0.6` (Lc0), `b = 0.25` (Stockfish), `c = 0.15` (result). Is that OK, or should the main net use Lc0 only at first to simplify Phase 4?
- **OpenBench:** self-host it (helps SPSA and datagen management), or stay with local fastchess scripts?
