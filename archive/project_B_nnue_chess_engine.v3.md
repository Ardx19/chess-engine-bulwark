# Project B: Defensive NNUE Chess Engine (plan v3)

## Context for the new chat

- **Who:** Final-year/fresher candidate targeting SDE and AI/ML roles. Strong in C++ (research internship: RDKit batch featurization with OpenMP and Boost.Python), Go (distributed social feed engine), applied ML (RNA classification, agentic chemistry toolchain), and AWS (Amazon HackOn project).
- **Goal:** Add breadth to the resume with advanced, measurable projects. Resume advice already given: keep 4 projects, order by strength, add a skills section, fix LaTeX issues (stray `\end{contained}`, empty `\href{}`), and drop the chess project already commented out in favor of the new engine.
- **Decisions made:**
  1. Project A: a transactional sandbox for coding agents, extended into an **agent eval harness**. Kept as the headline project.
  2. Project B: a **bitboard C++ chess engine with a trained NNUE evaluation**, now with a clear identity: **a defensive specialist**. It recognizes fortresses, steers lost positions toward draws (repetition, stalemate, 50-move rule, tablebase draws), and proves it with a defense benchmark.
  3. The net is **big in capacity but sparse in compute**: king-bucketed input layer, WDL (win/draw/loss) output, and a **mixture-of-experts** experiment after the accumulator.
  4. **Chess960 / DFRC supported** from day 1.
  5. Dropped for now: mini LLM inference engine, Raft KV store, LoRA fine-tuning project.
- **Hardware:** 8 CPUs locally (WSL2, Linux 5.15) for engine dev, SPRT testing and data generation. Training runs on **Kaggle free-tier GPUs** (P100 or 2x T4, about 30 GPU-hours/week, 12-hour sessions), with **Colab free tier as fallback**. So training must be **checkpointed and resumable across platforms** (B.9).
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
- **Match runner:** fastchess (pentanomial SPRT). The 8-CPU budget is **7 concurrent games**, single-threaded engines, Hash 16-32 MB. **Never run SPRT and datagen at the same time** (CPU contention means time losses and noise): SPRT by day, datagen overnight.
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

## B.7 Labels, WDL targets and teacher policy

- **Game result** (W/D/L): noisy but carries draw information. **Search score**: informative but only gives an expected score, not a draw probability.
- **Scalar nets (v1):** `target = lambda*sigmoid(score/400) + (1 - lambda)*result`, lambda 0.6-1.0.
- **WDL nets (v2+):** loss `= CE(softmax(z), t) + a*((W + D/2) - sigmoid(score/400))^2`. `t` is the one-hot game result, **replaced by the exact tablebase WDL** whenever the position has <= 5 pieces. Use 50-move semantics: cursed win and blessed loss both count as a draw.
- **Tablebase rescoring and adjudication in datagen:** once a game reaches <= 5 pieces, end it with the tablebase result. That is cheaper and gives exact labels.
- **Data tracks** (reported separately): **own data** (the headline: bootstrapped from the HCE, then from each net) and **teacher data** (ablation only: the Lichess evaluation DB, CC0, about 416M Stockfish-evaluated positions, no game results, so it fits scalar or expected-score targets only).

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
More parameters need more data: N1 needs about 50-100M positions; N2 and N3 want **1B+**. Bucketed and MoE nets spread data thinner, which is another reason to use upcycling.

---

## B.9 Data generation and training on 8 CPUs + Kaggle/Colab

### Datagen (local, 8 CPUs)
- **Openings mix** (itself an ablation): standard start + 8-9 random plies; **Chess960/DFRC starts** + random plies (diversity); **UHO-style unbalanced positions** (lots of defense practice for the losing side). Discard lopsided starts (|eval| > ~1000 cp).
- Play with a soft limit of about 5k nodes/move at **neutral `d = 0.5`**. Adjudicate wins (|score| >= ~2000 for several plies) and draws. **End games at <= 5 pieces with the tablebase result.**
- Record: packed position, score (side-to-move POV), WDL result (tablebase-rescored), `rule50`. Filter at load time: in check, capture or promotion best move, extreme scores. **Weight** "drawn but imbalanced" positions as an ablation.
- Format: fixed-size packed record of about 32 bytes, bulletformat-compatible. Shards of about 1 GB (about 31M records).
- **Throughput budget:** 7 threads at about 150-200 positions/s each is about 4-5M positions/hour, so **about 40M per night** (10 h) and about 100M per full weekend day. 1B positions takes about 4-5 weeks of nights. **Start continuous datagen in Phase 4.** Bigger nets lower datagen NPS, so the datagen net is a deliberate choice (current best vs. faster).

### Where everything lives
| Item | Store | Why |
|---|---|---|
| Training data shards | **Private Kaggle Dataset** `nnue-data`, versioned per generation (`kaggle datasets version -p shards/ -m "gen2 +200M"`) | Attaches read-only at `/kaggle/input/...`. Colab pulls the same data with the Kaggle API. Check the current per-dataset size limit. |
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
- **Resume:** `train.py --resume auto` searches `/kaggle/input/*/ckpt`, then Drive, then local, and refuses a config-hash mismatch unless `--allow-config-change` is passed (for intended LR or fine-tune changes).
- **Deterministic data order:** the shard permutation comes from the seed, so the cursor alone determines what comes next. Resume reproduces the same stream.
- **Export:** `export.py <ckpt>` writes `net-<sha8>.nnue` (quantized) plus `ref-<sha8>.jsonl` (float outputs on 10k verification FENs). The engine's `verifynet` command compares against it. Registry row: net id, parent checkpoint, data, architecture, val loss, ECE, fixed-node and fixed-time Elo, defense benchmark.
- **Loader:** a C++ (pybind11) decoder from packed records to feature indices, because Kaggle's few CPU cores will bottleneck any Python parsing. Pin memory and prefetch batches.
- **Reference trainer:** bullet (Rust; CUDA, ROCm or Metal) on Kaggle for N1 only, to cross-check your PyTorch trainer.

---

## B.10 Phased plan (about 20 weeks of B time; cut lines marked)

| Phase | Weeks | Content | Exit |
|---|---|---|---|
| 0 | 1 (start) | Git, CMake (Release, Debug + sanitizers, portable), test runner | Builds and tests run |
| 1 | 1-3.5 | B.3 including Chess960: magics, staged pseudo-legal generator, copy-make, Zobrist, draws, FEN/X-FEN/Shredder-FEN | Perft gate (standard + 960 sample) |
| 2 | 4-5 | UCI (+ `UCI_Chess960`, Hash, Threads, Move Overhead), ID + PVS + qsearch + TT, PeSTO, basic time management, `bench`, fastchess SPRT pipeline | 1000 self-play games with no illegal moves or time losses; first SPRT passes |
| 3 | 5-8 | Selectivity ladder (at least steps 1-9), time-management upgrades, HCE + Texel, **Syzygy probing (D4)**, **rule50 scaling (D3)**, **qsearch stalemate guard (D5)** | **Fallback-complete:** Elo-per-feature table, anchor gauntlet, README |
| 4 | 9-10 | Datagen (opening mix, tablebase adjudication), data on a Kaggle Dataset, PyTorch trainer + C++ loader + **checkpoint/resume**, N1 float inference | N1 beats HCE at fixed nodes; a resume test (kill mid-run, resume, identical loss curve) passes |
| 5 | 11-12 | Quantized SIMD inference, SPRT vs HCE, retune margins; **continuous datagen starts** | +Y Elo over HCE; gauntlet |
| 6 | 13-15 | **Defense:** N2 (king buckets + WDL head), D1-D2 utility, D6-D7, defense benchmark built and run, calibration | Save-rate gain with the strength guard passing |
| 7 | 16-18 | **MoE:** N3 variants A/B/C (+ D), upcycling, specialization analysis | Equal-compute comparison table |
| 8 | 19-20 | Bootstrap generations, final gauntlets, write-up (results page + blog-style README) | Resume line filled with real numbers |

**Cut order if behind:** (1) Phase 7 shrinks to A vs B only (hard stacks) or moves after Project A ships; (2) D6-D8; (3) ladder steps 13-15; (4) Lazy SMP is never in scope. **Do not cut:** the perft gate, testing infrastructure, the WDL head, or the defense benchmark, because the specialty claim depends on them.

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
| Own data vs Lichess-Stockfish data | Size, architecture | Distillation vs self-play |
| Own PyTorch trainer vs bullet (N1) | Data, architecture | Trainer correctness |

## B.12 Risks

- **Specialization costs Elo:** large `d` or aggressive draw-seeking throws away wins. Hence a situational `d`, the strength guard, and neutral-`d` datagen.
- **Scope creep:** about 20 weeks against a 25-week shared budget. Follow the cut order; the engine is complete at Phase 3.
- **MoE instability:** routing flips cause eval jumps and search instability. Measure them; fall back to piece-count stacks.
- **Data starvation for big nets:** 1B+ positions needs weeks of nights. Upcycle, and start datagen early.
- **Cloud fragility:** quotas, 12-hour kills, Colab disconnects. Deadline-aware checkpoints and deterministic resume; the training run is tested for resume in Phase 4.
- **CPU contention:** never mix SPRT and datagen.
- **Movegen and 960 bugs poison data:** the perft gate includes the 960 suite.
- **Silent search bugs** (mate scores, draw scores under contempt, TT staleness when `d` changes): mate-in-N tests, `bench` determinism, asserts.
- **Self-play inflation:** anchor gauntlets.
- **Crowded space:** differentiated now by the defense identity + a measured benchmark + the MoE experiment + rigor.

## B.13 Resume target

"Defense-specialized chess engine in C++ (magic bitboards, Chess960, perft-verified on 950+ positions, X Mnps), PVS search with LMR, NMP, singular extensions, history heuristics and Syzygy probing; every change SPRT-tested. Trained a WDL NNUE (king-bucketed, ~12M params, int16/int8 AVX2) from scratch on M positions of self-play data with a fault-tolerant PyTorch pipeline on Kaggle/Colab. Raised save rate from losing positions by X pts at equal Elo; calibrated draw prediction (ECE e); learned-router mixture-of-experts vs piece-count layer stacks at equal compute: Δ Elo. ~Z CCRL-equivalent."

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

---

## Continue in a new chat

Paste this whole file as the first message, then try:
1. "Give me the Phase 0-1 checklist: repo layout, CMake targets, Position struct with Chess960 castling, magic finder, perft + 960 harness."
2. "Design the Phase 2 testing pipeline: bench positions (standard + 960), fastchess SPRT scripts, results log."
3. "Write the search()/qsearch() skeleton with the search stack, TT, mate scores, and hooks for the draw-value utility and tablebase probes."
4. "Design the datagen record format, the Kaggle Dataset layout, and train.py with the checkpoint/resume spec from B.9."
5. "Design the WDL head loss, the N2 king-bucket layout, and the N3 router + load-balancing loss in PyTorch, plus the C++ inference."

Open items:
- **CPU model:** AVX2 / BMI2 support (affects the PEXT build and SIMD paths). Are the 8 CPUs 8 cores or 8 threads (affects datagen and SPRT concurrency)?
- **Disk:** about 40 GB free for 1B positions + 1 GB for the 5-piece Syzygy tables? (6-piece needs 149 GiB, optional.)
- **Reference opponent** for the save-rate benchmark: which engine and time control?
- **Teacher policy:** own data only for the headline (recommended), with the Lichess-Stockfish data as an ablation?
- **OpenBench:** self-host it (helps SPSA and datagen management), or stay with local fastchess scripts?
