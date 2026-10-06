# Project B: Bitboard Chess Engine with NNUE (plan v2)

## Context for the new chat

- **Who:** Final-year/fresher candidate targeting SDE and AI/ML roles. Strong in C++ (research internship: RDKit batch featurization with OpenMP and Boost.Python), Go (distributed social feed engine), applied ML (RNA classification, agentic chemistry toolchain), and AWS (Amazon HackOn project).
- **Goal:** Add breadth to the resume with advanced, measurable projects. Resume advice already given: keep 4 projects, order by strength, add a skills section, fix LaTeX issues (stray `\end{contained}`, empty `\href{}`), and drop the chess project already commented out in favor of the new engine.
- **Decisions made:**
  1. Project A: a transactional sandbox for coding agents, extended into an **agent eval harness** the user will genuinely use for their own agent work. Kept as the headline project.
  2. Project B: a **bitboard C++ chess engine with a trained NNUE evaluation**. Engine first, net second.
  3. Dropped for now: mini LLM inference engine, Raft KV store, LoRA fine-tuning project. They remain options later.
- **Sequencing:** about 25 weeks of work in total at 10-15 hours per week, so overlap them. Sandbox Phase 0 comes first because the harness depends on it. Chess phases fill gaps while API runs or match batches are in progress. The chess engine is resume-complete after **Phase 3 (the tested handcrafted engine)**, so it is the fallback if the harness runs late.
- **Dev environment:** WSL2 (Linux 5.15) on Windows. Keep source, binaries and datasets on the Linux filesystem (`~/...`), not `/mnt/c`, because cross-filesystem I/O is slow. PyTorch can use an NVIDIA GPU through CUDA on WSL2 if one is present.
- **This file covers Project B only.** Project A (sandbox and eval harness) has its own file. The previous version of this plan is saved as `project_B_nnue_chess_engine.v1.md`.

### What changed from v1, and why

| Change | Why |
|---|---|
| UCI, `bench`, and SPRT testing moved from week 7 to **week 4**, before any search feature | v1 said "keep only builds that gain Elo" in Phase 2 but built the tool that measures Elo in Phase 3. Chessprogramming.org's *Getting Started* puts UCI, time management, and SPRT self-play before search work. |
| Board-representation decisions are made explicitly up front (B.3) | Mapping, piece storage, copy-make vs make/unmake, legality strategy and Chess960 castling are expensive to change later. |
| Search features follow the community "Connorpasta" order, each one SPRT-gated (B.4) | That list covers most of the Elo found in top engines, in an order where each step builds on the previous one. |
| The first net is `(768 -> N)x2 -> 1` with **no hidden layers**, not 768-256-32-1 | One quantized layer is easy to get exactly right. Hidden layers only pay off with much more data and need int8 SIMD work. This is the standard starting point (bullet's `simple` example). |
| Datagen uses a **node limit (~5k nodes) with random opening plies**, not fixed depth 6-8 | Gives more uniform cost per position, more diverse data, and is standard practice. |
| Data-size guidance raised: first nets want tens to hundreds of millions of positions | 5-20M positions usually gives a net that struggles to beat a good handcrafted eval. Throughput math is in B.9. |
| The Texel tuner and the NNUE trainer **share one data format and one loss** | Texel tuning is logistic regression on handcrafted features. NNUE is the nonlinear version of the same thing, so one pipeline serves both. |
| **Fixed-node vs fixed-time Elo** are reported separately | This separates eval quality from eval speed, which is the core tradeoff of NNUE. |
| Search margins are **retuned after switching to NNUE** | Pruning margins assume an eval scale. A new eval changes that scale. |
| Teacher data: the Lichess evaluation DB (CC0) replaces "check Stockfish's GPL" | It removes the licensing question for the teacher ablation. Own-data generations stay the headline. |

---

## B.1 Goal and success criteria

A UCI-compliant, bitboard-based C++ chess engine with a tuned search and an NNUE evaluation trained by the user. Every change is validated by SPRT, and strength is reported against calibrated anchors.

| Milestone | Measurable outcome | Rough Elo prior (CCRL Blitz scale; replace it with your anchor-gauntlet number) |
|---|---|---|
| Phase 1: movegen | Perft exact on the 6 CPW positions plus an EPD suite of 100+ positions; perft Mnps reported | n/a |
| Phase 3: HCE engine | Elo-per-feature SPRT table, anchor gauntlet rating | ~2200-2600 |
| Phase 5: first NNUE | +Y Elo over HCE (SPRT), at fixed time **and** at fixed nodes | ~2700-3100 |
| Phase 6: generations | Elo-per-generation curve, 2+ ablations | 3200+ is a stretch that usually needs months of tuning and much more data |

Self-play Elo gains overstate gains against other engines. The anchor gauntlet (B.6) gives the honest number for the resume.

## B.2 Mental model: three layers and where they couple

1. **Board representation and move generation:** correctness first, speed second. A bug here silently poisons search, testing, and training data.
2. **Search:** makes the static eval more accurate by looking ahead. Almost all modern search strength comes from *selectivity* (reductions and pruning) driven by good move ordering.
3. **Evaluation:** `eval(position)` is one replaceable function called at the leaves (and in pruning decisions). HCE first, then NNUE.

Where the layers couple (most v1 risks live here):
- **Eval scale ↔ pruning margins:** RFP, futility, razoring, aspiration deltas and SEE thresholds are all in centipawns. A new eval changes that scale, so these values must be retuned.
- **Eval speed ↔ depth:** a slower but better eval can lose Elo at fixed time. Measure both fixed-node and fixed-time results.
- **Movegen correctness ↔ data quality:** illegal or missing moves corrupt every training label. This is why the perft gate exists.
- **Transposition table ↔ path-dependence:** mate scores need ply adjustment. Repetition and 50-move draws are path-dependent (graph-history interaction), so a TT entry can carry a draw score into a different path.

Key concepts kept from v1, with one correction:
- Win probability = `sigmoid(cp / K)`, with K around 400 and fit to data. v1 used `tanh(cp/K)` into [-1, 1]. That is equivalent up to scaling (`tanh(x/2) = 2*sigmoid(x) - 1`), but use sigmoid in [0, 1] to match Texel tuning and NNUE tooling.
- NNUE: sparse binary inputs, a wide first layer kept as a running sum (the accumulator), and tiny output layers. A quiet move changes 2 features, a capture 3, castling 4. It is value-only (no policy). Mate and stalemate come from search.

---

## B.3 Board representation: decisions to make before coding

Chessprogramming.org groups board representations as **piece-centric** (bitboards, piece lists), **square-centric** (8x8, 10x12 mailbox, 0x88) and **hybrid**. Modern engines are hybrid: bitboards for move generation, attacks and eval, plus a mailbox to answer "what piece is on square x?" in O(1), which bitboards are poor at.

| Decision | Choice | Reason / alternative |
|---|---|---|
| Square mapping | **LERF**: a1 = 0 ... h8 = 63, `sq = 8*rank + file`; north = `<< 8`, east = `<< 1` (mask out the H file). Constants: A-file `0x0101010101010101`, rank 1 `0xFF`. | The CPW standard, so every CPW snippet works unchanged. `sq ^ 56` flips ranks, which you need for black's perspective in PSTs and NNUE features. |
| Piece storage | **6 piece-type bitboards + 2 color bitboards** (occupancy derived), plus `Piece mailbox[64]` | Fewer bitboards to update than 12, and "all pawns" or "all sliders" queries are one AND. The 12-bitboard layout from v1 also works. Pick one and stay consistent. |
| Sliding attacks | **Fancy magic bitboards**: per-square mask, `(occ & mask) * magic >> shift`, then a lookup. About 800 KiB for rooks and 38 KiB for bishops (plain magics are about 2.3 MiB). Find magics yourself with random sparse numbers (`r1 & r2 & r3`), checked for collisions against every subset of relevant occupancy. | **PEXT (BMI2)** behind a compile flag is a clean extra: fast on Intel Haswell+ and AMD Zen 3+, but microcoded and very slow on Zen 1/2. Rotated bitboards and Hyperbola Quintessence are historical or educational alternatives. |
| Other tables | Knight, king and pawn attacks; `between[64][64]` and `line[64][64]` | `between` and `line` make pins, check blocks and legality tests one AND each. |
| Move encoding | **16-bit**: from (6), to (6), flags (4): quiet, double push, 2 castles, capture, en passant, 4 promotions, 4 promotion-captures | Cache-friendly. Move lists are a fixed `Move[256]` on the stack. |
| Make/unmake | **Copy-make** of a compact `Position` (about 150 bytes) into a ply-indexed stack | No unmake bugs, and NNUE accumulators are per-ply anyway. Alternative: make/unmake with a `StateInfo` undo stack. Benchmark both with perft if curious. |
| Legality | **Pseudo-legal generation + cheap `is_legal(move)`**, using `checkers` and `pinned` bitboards computed once per node. Only king moves, pinned pieces and en passant need real work. Use a dedicated evasion generator when in check (single check: capture, block or king move; double check: king moves only). | Fits staged move generation, since you generate captures first and often stop early. A fully legal generator (checkmask + pinmask) is a later perft-speed option. |
| Zobrist | 12x64 piece-square keys + side + 16 castling-state keys + 8 en passant file keys. **Hash the en passant file only if an enemy pawn can actually capture.** Also keep a pawn key and non-pawn keys. | Without the en passant rule, identical positions get different keys and repetitions are missed. Pawn and non-pawn keys feed the pawn hash and, later, correction history. Debug assert: incremental key == recomputed key after every move. |
| Castling rights | 4-bit mask; `rights &= castle_mask[from] & castle_mask[to]` | One line handles rook captures and king moves. |
| Draws | Repetition by scanning the key history back to the last irreversible move in steps of 2 plies; 50-move rule; insufficient material | Inside the search, treat a single repetition as a draw. Against game history, require threefold. |
| Chess960 / DFRC | **Decide now.** At minimum, encode castling internally as "king to rook-square" and store rook files. | DFRC openings are a standard source of datagen diversity, and retrofitting castling is painful. UCI_Chess960 itself can wait. |

**Perft gate (exit criterion for Phase 1).** All counts must match exactly before any search code exists:

| Position | FEN | Depth: nodes |
|---|---|---|
| Start | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` | 5: 4,865,609 / 6: 119,060,324 |
| Kiwipete | `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -` | 4: 4,085,603 / 5: 193,690,690 |
| 3 (en passant, rook endgame) | `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1` | 6: 11,030,083 |
| 4 (promotions, castling) | `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1` (plus its mirror) | 5: 15,833,292 |
| 5 (historic bug-catcher) | `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8` | 4: 2,103,487 |
| 6 (symmetric middlegame) | `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10` | 4: 3,894,594 |

Then run a larger EPD perft suite (100+ positions). Use `perft divide` against a reference engine to bisect mismatches. Keep all of it as a regression test. Report perft Mnps both **with bulk counting** (leaf count = move-list size) and **without**, because the first number flatters and the second is honest.

---

## B.4 Search: architecture and feature ladder

### Core framework (week 4; all of it must exist before SPRT is meaningful)
- **Negamax, fail-soft alpha-beta, PVS.** Template the node type (`PV` / `NonPV`). The first move gets the full window. Later moves get a null window `(-alpha-1, -alpha)` and are re-searched with the full window only if `alpha < score < beta` at a PV node.
- **Iterative deepening** with a **triangular PV table** for reliable `info ... pv` output.
- **Quiescence search:** stand-pat on static eval; search captures and promotions in MVV-LVA order. When in check, search all evasions (no stand-pat). Probe the TT here too.
- **Transposition table:** power-of-two size; entries hold key verification bits, move, score, static eval, depth, bound (exact/lower/upper) and age. Use buckets with depth-preferred replacement plus aging. **Mate scores are ply-adjusted on store and probe.** No TT cutoffs at PV nodes. Prefetch the entry after making a move.
- **Search stack** per ply: static eval, killers, current move, excluded move, continuation-history pointer. The heuristics below need it.
- Repetition and 50-move draws, **mate distance pruning**, **check extension**.

### Move ordering (staged move picker)
TT move, then good captures (SEE >= 0, ordered by MVV + capture history), then killers (2 per ply), then the countermove, then quiets by butterfly history + continuation history, then bad captures. Generate each stage lazily. A TT-move cutoff happens before any moves are generated (in Stockfish, about 75% of cutoffs at nodes that have a hash move come from that move). Implement SEE as a threshold test `see_ge(move, threshold)` (swap algorithm, adding x-ray attackers through the bitboards after each capture).

### Selectivity ladder: one step per SPRT, in this order
From CPW's *Search Progression* page ("Improved Connorpasta"). Starting values are common defaults to tune later, not truths.

| # | Feature | Gist / typical start |
|---|---|---|
| 1 | Aspiration windows | From depth ~4: window = previous score ± ~25 cp; widen the delta geometrically on fail. |
| 2 | Reverse futility pruning (static null move) | Non-PV, not in check, depth <= ~8: if `eval - 80*depth >= beta`, return eval. |
| 3 | Null move pruning | Non-PV, not in check, side has non-pawn material, `eval >= beta`, previous move not null: search null move at `depth - R`, with `R = 3 + depth/3` (+ an eval-margin term). Fail high, then cut. |
| 4 | Late move reductions | `R = 0.78 + ln(depth)*ln(moveNumber)/2.47` (Ethereal-style) for quiets. Reduce less at PV nodes, for killers and good history; reduce more at cut nodes. Re-search on fail high. |
| 5 | Butterfly history + killers | Bonus about depth², applied with the **history gravity** update `h += b - h*abs(b)/MAX`. **Malus** for quiets searched before the cutoff move. |
| 6 | Late move pruning + futility pruning | LMP: skip quiets after `3 + depth²` moves (fewer if not improving). Futility: at shallow depth, skip quiets if `eval + margin(depth) <= alpha`. |
| 7 | Internal iterative reductions (IIR) | No TT move at depth >= ~4: reduce depth by 1. Replaces IID. |
| 8 | "Improving" heuristic | Static eval > static eval 2 plies ago. Feeds RFP, LMP and LMR. |
| 9 | SEE pruning (qsearch and main search) | Skip captures with `!see_ge(move, -margin*depth)`. In qsearch, prune SEE < 0. |
| 10 | Continuation history (1-ply and 2-ply) | Indexed by [previous piece, previous to-square][piece, to-square]. Use it in ordering and LMR. |
| 11 | Capture history | Indexed by [piece][to][captured type]. |
| 12 | History pruning / history-adjusted LMR | Prune or reduce quiets with very negative history at low depth. |
| 13 | Singular extensions + multicut + double/negative extensions | If the TT move is a lower bound at depth >= depth-3, search the other moves with `beta_s = ttScore - k*depth` at half depth, excluding the TT move. If all fail low, extend the TT move. If one fails high above beta, multicut. |
| 14 | Static eval correction history | Tables keyed by pawn, material and non-pawn hashes store an EMA of `searchScore - staticEval` and correct future evals. Introduced by Caissa in 2023; gains grow at longer time controls. |
| 15 | Qsearch futility / delta pruning | Skip captures that cannot raise alpha even with the captured piece's value plus a margin. |

After the ladder: tune the search constants with **SPSA** (OpenBench supports it), then repeat after the NNUE switch.

### Time management
- **Soft and hard limits.** Start with `soft = remaining/20 + increment/2` (CPW: competitive with much fancier schemes). Make the hard limit a few times the soft limit, capped well below the remaining time, minus a `Move Overhead` UCI option.
- Check the soft limit between iterations; don't start a new iteration if most of the soft budget is gone. Check the hard limit every N nodes inside search.
- Upgrades (SPRT each): **node-based scaling**, so the soft limit grows when only a small fraction of nodes went to the best move; best-move stability; eval stability.

### Parallel search (stretch only)
Lazy SMP: threads share the TT and search the same root at staggered depths, and scales well to about 8 or more cores. It is not needed for testing or datagen, which both run many single-threaded games in parallel. Lockless TT entries (XOR the key with the data) are required.

---

## B.5 Handcrafted evaluation: small on purpose

The HCE has two jobs: be the **Elo baseline** for the NNUE delta, and **generate gen-0 training data**. Timebox it, because the net replaces it.

- **Tapered eval:** packed mg/eg scores in one int32. Phase weights N = B = 1, R = 2, Q = 4, total 24: `eval = (mg*phase + eg*(24 - phase)) / 24`.
- **Baseline:** PeSTO material and piece-square tables (on CPW) give a strong starting point for almost no work.
- **Optional terms**, each SPRT'd: mobility, passed pawns, doubled and isolated pawns, bishop pair, simple king safety. Cache pawn terms in a pawn hash table.
- **Texel tuning, built as the NNUE trainer's first model.** Minimize `E = mean((R - sigmoid(K*q/400))^2)`, where q is the quiet (qsearch) score and R ∈ {0, 0.5, 1}. Fit K once, then optimize the parameters with Adam in PyTorch, with the eval written as a linear model over feature counts. Data: your own self-play positions in the B.9 format, or the public Zurichess quiet-labeled set. This builds the data loader, the loss and the train/validation split that the NNUE trainer reuses.
- **Symmetry test:** `eval(pos) == eval(color_flip(pos))` on thousands of FENs. It catches PST indexing bugs instantly, and it applies to the net too.

---

## B.6 Testing and measurement infrastructure (built in week 4)

- **Git from day 1.** This directory is not a git repo yet. Every functional commit message carries `Bench: <nodes>`.
- **`bench` command:** fixed-depth search over about 50 built-in FENs. Prints total nodes and NPS. It is the determinism signature: refactors must keep the node count identical, and a changed count means behavior changed.
- **Correctness suite:** perft (B.3), FEN round-trip, incremental-vs-recomputed Zobrist, SEE unit tests, eval symmetry, a few mate-in-N positions. Debug builds use asserts and `-fsanitize=address,undefined`.
- **Match runner:** **fastchess** (pentanomial SPRT, which finishes faster than trinomial); cutechess-cli is the alternative. Engines run single-threaded, Hash 16-32 MB, concurrency = cores - 1.
- **Time controls:** STC 8+0.08 for most tests. LTC 40+0.4 for search changes that may scale differently with depth.
- **Opening books:** `8moves_v3.pgn` while the engine is weak. Switch to `UHO_Lichess_4852_v1.epd` when draw rates rise. Each opening is played twice with colors reversed.
- **SPRT bounds** (alpha = beta = 0.05, so LLR bounds ±2.94). Per the CPW table:

  | Engine strength | Gainer | Non-regression / simplification |
  |---|---|---|
  | Weak (early) | [0, 10] | [-10, 0] |
  | Top-200 level | [0, 5] | [-5, 0] |

- **Optional:** self-host **OpenBench**. It manages SPRT queues, SPSA tuning and distributed datagen, and makes a strong resume talking point.
- **Anchor gauntlet** at the end of Phases 3, 5 and 6: 1000+ games against 3-4 open-source engines with known CCRL ratings near your level. This turns inflated self-play Elo into an honest CCRL-equivalent estimate.
- **Report every result** as Elo ± 95% error, games, LLR, bounds, time control and book.
- A Lichess bot is optional public proof, but its rating is on a different scale. Don't mix it with CCRL numbers.

---

## B.7 Labels and teacher policy

Every label is a proxy:
- **Game result** (1, 0.5, 0 from the side to move's view): unbiased but noisy.
- **Search score** mapped through `sigmoid(cp/400)`: much more information per position, but bounded by the engine that produced it.
- **Blend:** `target = lambda*sigmoid(score/400) + (1 - lambda)*result`. Common starting range: lambda 0.6-1.0, meaning a WDL weight of 0-0.4.

Two data tracks, always reported separately:
- **Own data (headline): self-bootstrapping.** Gen 0 is HCE self-play. Train v1, generate data with v1, train v2, and so on. It is slower but fully yours, and it produces the Elo-per-generation curve. Parts of the engine community (some rating lists and tournaments) distinguish engines trained on original data, so keep this track pure.
- **Teacher data (ablation): the Lichess evaluation database.** About 400M+ Stockfish-evaluated positions (FEN, depth, cp or mate), released under **CC0**, so the GPL question disappears. Caveats: it has **no game results** (lambda = 1 only), and the positions come from human analysis, a different distribution from self-play. The ablation "teacher data vs own data at equal size" is a strong experiment. Running Stockfish locally as a labeler also works: its output is generally not GPL-covered, but its code or nets would be. Disclose any teacher in the README.

---

## B.8 NNUE design

### Starting architecture
`(768 -> N)x2 -> 1`, **SCReLU** (`clamp(x, 0, 1)^2`), perspective accumulators (side-to-move half first), **no hidden layers**. Use N = 128 for the first pipeline test, then 256.

- **Feature index** for perspective `p`: `relative_color*384 + piece_type*64 + (p == WHITE ? sq : sq ^ 56)`.
- **Quantization** (bullet-style defaults): accumulator weights and biases × QA = 255 (int16); output weights × QB = 64 (int16); `SCALE = 400`. Output: `(sum(SCReLU(acc_i) * w_i) / QA + bias) * SCALE / (QA * QB)`.
- **SCReLU SIMD trick:** clip v to [0, QA], compute `(v*w)` in int16 first, then `madd` with v into int32. This needs `|w_quant| <= 127`, so **clamp float weights to ±1.98 during training**.
- **Accumulator engineering:** an accumulator stack indexed by ply, filled by copy-make. **Lazy updates:** record the feature deltas on make, and materialize only when eval is called. Fused add/subtract. AVX2 path with a scalar fallback.

### Upgrade ladder (one at a time; SPRT each, at fixed nodes and at fixed time)
1. Wider L1 (256 → 512 → 1024), with data scaled up alongside.
2. **Output buckets:** 8 sets of output weights selected by `(piece_count - 2) / 4`. Cheap and usually a gain.
3. **Horizontal mirroring + king input buckets** (HalfKA-style). Crossing a bucket boundary forces an accumulator refresh, so add **Finny tables** (accumulator refresh caches).
4. Hidden layers (L1 → 16 → 32 → 1, int8). Stretch goal; only worth it with hundreds of millions of positions.

Rules of thumb: first-layer width matters more than depth, bigger nets need more data, and an eval that is 10% better but 3x slower usually loses Elo. Always measure.

---

## B.9 Data generation and training pipeline

### Datagen (C++ `datagen` subcommand, many single-threaded games in parallel)
- **Openings:** 8-9 random legal plies from the start position (or DFRC starts if supported). Do a short verification search and discard openings that are already lopsided (|eval| > ~1000 cp).
- **Play:** soft node limit around 5,000 per move, with a small hash.
- **Adjudication:** win when |score| >= ~2000 for several consecutive plies; draw on repetition, 50-move rule, insufficient material, or a long run of near-zero evals.
- **Record** for every position: packed position, score from the side to move's view, final WDL result.
- **Filter at load time, not at generation time:** skip in-check positions, positions whose best move is a capture or promotion (noisy), and mate or extreme scores. This keeps ablations over filters cheap.
- **Format:** a fixed-size packed record of about 32 bytes (occupancy bitboard + 4-bit piece codes + score + result). Making it **bulletformat-compatible** lets you use bullet as a reference trainer.
- **Throughput math** (measure your own): 5k nodes/move at about 1M NPS/thread gives about 150-200 positions/s/thread. On 8 threads that is about 5M positions/hour, or about 100M/day. Sizes: first net 50-100M positions; serious generations 300M+.

### Trainer
- **Own PyTorch trainer** (the AI/ML resume value). A C++ loader (pybind11 or ctypes) decodes packed records straight into feature-index tensors. No FEN parsing in Python.
- Sparse first layer: `EmbeddingBag(mode="sum")` or sparse matmul. Two perspectives share weights.
- **Optimizer:** AdamW, learning rate about 1e-3 with cosine or step decay, batch 16,384, weight clamping (above), shuffled interleaving across files.
- **Loss:** `(sigmoid(out/400) - target)^2`. Validation set held out **by game**, to avoid leakage.
- **Reference baseline:** train the same architecture on the same data with **bullet** (Rust, the trainer most top engines use). If your net is clearly weaker, debug your trainer. "Own trainer vs bullet" is also a reportable result.

### Deployment
Export the weights. First run a **float C++ forward pass** and match PyTorch on 10k positions to within float rounding. Then run the **quantized** path and match the float path within quantization tolerance. Only then measure NPS and Elo.

---

## B.10 Phased plan (about 15 weeks of Project B time)

**Phase 0 (start of week 1): skeleton.** Git repo, CMake with Release (`-O3 -march=native`), Debug + sanitizers, and a portable build. Test runner. FEN parse and print.

**Phase 1 (weeks 1-3): board and move generation.**
- B.3 decisions; magic finder; attack tables; pseudo-legal staged generator with `is_legal`; copy-make; Zobrist with debug verification; draw detection.
- **Exit:** the full perft gate passes, perft Mnps reported both ways, and the Zobrist and FEN tests pass.

**Phase 2 (week 4): minimum playable engine + test infrastructure.**
- UCI (`uci`, `isready`, `ucinewgame`, `position`, `go` with all time-control parameters, `stop`; options Hash, Threads, Move Overhead); `info` lines with depth, seldepth, score cp/mate, nodes, nps, hashfull and pv.
- Search: ID + PVS + qsearch + TT + MVV-LVA. Eval: material + PeSTO PSTs. Basic time management. `bench`.
- fastchess SPRT script, opening book, results log.
- **Exit:** 1,000 games at 8+0.08 against itself with zero illegal moves, crashes or time losses. The first SPRT (e.g. TT on vs off) passes.

**Phase 3 (weeks 5-8): search ladder + HCE. This is the fallback-complete milestone.**
- Work down the B.4 ladder through step 9 or further, then time-management upgrades, then the HCE terms and Texel tuner from B.5. One SPRT per change; log every result, including failures.
- **Cut line if behind:** stop after step 9 (SEE pruning). The remaining steps add smaller gains.
- **Exit:** Elo-per-feature table, anchor gauntlet rating, README with methodology. **The engine is resume-ready here.**

**Phase 4 (weeks 9-10): data + first net.**
- Datagen subcommand; gen-0 data from the HCE engine (start at 20-50M positions, keep generating in the background).
- PyTorch trainer + C++ loader; train `(768 -> 128)x2 -> 1`; float C++ inference; verify against PyTorch.
- **Exit:** the float net beats HCE at **fixed nodes** (proving eval quality before any speed work).

**Phase 5 (weeks 11-12): make it fast.**
- Quantization, incremental lazy accumulators, AVX2 SCReLU; 256-wide net on more data.
- SPRT NNUE vs HCE at fixed time. **Retune the search margins** (RFP, futility, aspiration, SEE) for the new eval scale; SPSA if OpenBench is set up.
- **Exit:** +Y Elo over HCE (SPRT), NPS comparison, anchor gauntlet.

**Phase 6 (weeks 13-15): own experiments.**
- Bootstrap generations (v1 → v2 → v3 data and nets) with the Elo-per-generation curve.
- Two or three ablations from B.11. Write the results page.

## B.11 Experiment menu (pick 2-3)

| Experiment | Holds fixed | What it shows |
|---|---|---|
| Elo per bootstrap generation | Architecture, data size | Self-improvement without a teacher |
| L1 width (128/256/512/1024) | Data | Accuracy vs inference-cost tradeoff (report fixed-node *and* fixed-time Elo) |
| Data size scaling (25/50/100/200M) | Architecture | Data-limited vs capacity-limited regime |
| Lambda / WDL weight sweep | Data, architecture | Value of search scores vs game results as labels |
| Output buckets on/off | Everything else | Cheap feature-engineering gain |
| King buckets + mirroring | Everything else | HalfKA-style inputs vs refresh cost |
| Own data vs Lichess-Stockfish data | Size, architecture | Distillation vs self-play |
| Own PyTorch trainer vs bullet | Data, architecture | Trainer correctness and performance |
| Position filters (in check, capture best move) | Data source | Label-noise effects |

## B.12 Risks

- **Crowded space:** many students build chess engines. Differentiate with rigor (perft-verified, SPRT on every change, anchor-calibrated ratings) and a real experiment with ablations.
- **Testing late = untested features:** fixed by building the test infrastructure in week 4.
- **Movegen bugs poison data:** perft gate, hash verification, symmetry tests.
- **Silent search bugs:** mate-score TT bugs and draw or repetition errors can still pass SPRT. Use mate-in-N tests, a `bench` determinism check, and debug asserts.
- **Self-play inflation:** gains against yourself shrink against other engines, hence the anchor gauntlets.
- **Eval-scale drift after the NNUE switch:** retune the margins (B.4).
- **Time sink in tuning:** each phase has a cut line. HCE work is deliberately capped.
- **Compute:** datagen is CPU-bound (budget overnight runs); training needs a GPU (local CUDA on WSL2, or Kaggle/Colab). Keep data on the Linux filesystem.
- **Faster is not always better:** always confirm with matches, at both fixed nodes and fixed time.
- **Resume overlap:** replace the earlier chess project; do not list both.

## B.13 Resume target

"Bitboard chess engine in C++ (magic bitboards; perft-verified on 100+ positions; X Mnps movegen), UCI-compliant, PVS search with null-move, LMR, singular extensions and history heuristics, every change SPRT-tested (N tests). Trained an NNUE evaluation from scratch via self-play bootstrapping on M million positions with a custom PyTorch trainer and AVX2 int16 inference, gaining +Y ± e Elo over handcrafted eval; ~Z CCRL-equivalent vs calibrated anchors; ablations on network width, data scale and label mix."

## B.14 Reading list (chessprogramming.org unless noted)

- Board: *Board Representation*, *Bitboards*, *Square Mapping Considerations*, *Magic Bitboards*, *BMI2*, *Move Generation*, *Zobrist Hashing*, *Repetitions*, *Perft Results*.
- Search: *Getting Started*, *Search Progression*, *Principal Variation Search*, *Transposition Table*, *Move Ordering*, *Quiescence Search*, *Static Exchange Evaluation*, *Null Move Pruning*, *Late Move Reductions*, *Futility Pruning*, *History Heuristic*, *Internal Iterative Reductions*, *Singular Extensions*, *Static Evaluation Correction History*, *Time Management*, *Lazy SMP*.
- Eval and training: *Tapered Eval*, *PeSTO's Evaluation Function*, *Texel's Tuning Method*, *NNUE*, *Stockfish NNUE*.
- Testing: *Engine Testing*, *Sequential Probability Ratio Test*.
- External: bullet trainer (github.com/jw1912/bullet, start from `examples/simple`), nnue-pytorch (official-stockfish), OpenBench (github.com/AndyGrant/OpenBench), fastchess, Lichess database (database.lichess.org), and the Engine Programming Discord.

---

## Continue in a new chat

Paste this whole file as the first message, then try:
1. "Give me a detailed checklist for Phase 0-1: repo layout, CMake targets, Position struct, magic finder, and the perft/EPD test harness."
2. "Design the Phase 2 testing pipeline: bench positions, fastchess SPRT script, results log format."
3. "Write the search() and qsearch() skeleton for Phase 2 with the search stack, TT and mate-score handling, ready for the ladder."
4. "Design the packed data record, the datagen loop, and the PyTorch trainer + C++ loader for Phase 4."

Open items:
- **GPU:** is there a local NVIDIA GPU usable via CUDA on WSL2, or will training run on Kaggle/Colab?
- **CPU:** core count, and AVX2 / BMI2 support (affects datagen throughput and whether a PEXT build is worth it; avoid PEXT on AMD Zen 1/2).
- **Teacher policy:** own-data-only headline (recommended), with the Lichess-Stockfish data as an ablation only?
- **Chess960 castling:** support it internally from day 1 (recommended) or skip?
- **OpenBench:** self-host it, or stay with local fastchess scripts?
