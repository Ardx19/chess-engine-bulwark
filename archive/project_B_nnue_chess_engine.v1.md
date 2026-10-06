# Project B: Bitboard Chess Engine with NNUE

## Context for the new chat

- **Who:** Final-year/fresher candidate targeting SDE and AI/ML roles. Strong in C++ (research internship: RDKit batch featurization with OpenMP and Boost.Python), Go (distributed social feed engine), applied ML (RNA classification, agentic chemistry toolchain), and AWS (Amazon HackOn project).
- **Goal:** Add breadth to the resume with advanced, measurable projects. Resume advice already given: keep 4 projects, order by strength, add a skills section, fix LaTeX issues (stray `\end{contained}`, empty `\href{}`), and drop the chess project already commented out in favor of the new engine.
- **Decisions made:**
  1. Project A: a transactional sandbox for coding agents, extended into an **agent eval harness** the user will genuinely use for their own agent work. Kept as the headline project.
  2. Project B: a **bitboard C++ chess engine with a trained NNUE evaluation**. Engine first, net second.
  3. Dropped for now: mini LLM inference engine, Raft KV store, LoRA fine-tuning project. They remain options later.
- **Sequencing:** about 25 weeks of work in total at 10-15 hours per week, so overlap them. Sandbox Phase 0 comes first because the harness depends on it. Chess phases 1-3 fill gaps while API runs or match batches are in progress. The chess engine is complete after its Phase 3, so it is the fallback if the harness runs late.
- **This file covers Project B only.** Project A (sandbox and eval harness) has its own file.

---

## B.1 Goal

A UCI-compliant, bitboard-based C++ chess engine with a tuned search and a neural evaluation (NNUE) trained by the user, with every change validated by SPRT testing. Realistic target for a solo fresher: roughly 2800-3200 Elo against calibrated anchors, with a measured gain from the net over the handcrafted evaluation.

## B.2 Key concepts

- **Position evaluation:** the true value (win, draw, loss under perfect play) is unknowable, so you approximate it with v(S). Centipawn scores map to [-1, 1] with tanh(cp / K), K around 400, fit to data. Move quality is v(after move) minus v(before), computed with search.
- **Search refines evaluation:** minimax or alpha-beta to depth d gives a better estimate than the static eval. Quiescence search avoids evaluating mid-capture positions.
- **NNUE (Efficiently Updatable Neural Network):** sparse binary input (piece, square, optionally relative to the king), a wide first layer maintained as a running sum called the accumulator, then a few tiny dense layers. A move changes only 2-4 features, so the accumulator is updated by subtracting and adding weight columns instead of recomputing. It outputs a single score from the side to move's perspective. It is value-only: no move or policy output. Mate and stalemate scores come from the search, not the net.
- **The net is one replaceable function `eval(position)`** called at the leaves of the search. Strength comes from both search and eval.

## B.3 Labels, and the Stockfish question

There is no true label. Every label is a proxy:
- **Game result** (+1, 0, -1): unbiased but noisy.
- **Search score** from an engine, mapped through tanh: far more informative per position but only as good as the engine that made it.
- **Blend:** `target = lambda * score_target + (1 - lambda) * result`, with lambda around 0.7-1.0.

Two routes:
- **Distillation from Stockfish:** fast, strong, but not from scratch and capped near Stockfish's quality. Must be disclosed as a teacher in the README. Check Stockfish's license (GPL) for anything derived from its data or nets.
- **Self-bootstrapping:** handcrafted eval tuned with Texel tuning, self-play to generate data, train net v1, use v1 to generate better data, train v2, repeat. Slower, fully yours, and gives an Elo-per-generation curve.

Recommended hybrid: build the engine, train v1 (distillation allowed, clearly labeled), then run your own bootstrapped generations and experiments, and report them separately.

## B.4 Net design choices

- **Constraints, not a formula:** inference cost trades directly against search depth, and bigger nets need more data.
- **Starting architecture:** 768 inputs (12 piece types x 64 squares), first layer 256, then 32, then 1, clipped ReLU, evaluated from the side to move's perspective. Combine two perspective accumulators with the side to move first.
- **Scaling method:** get the pipeline working small, then double one thing at a time (first-layer width, second-layer width, data size). Track validation loss and Elo from real matches. Stop when the gain per doubling is small.
- **Width of the first layer matters more than depth.** Most NNUEs have only 2-3 layers after the accumulator.
- **Data guide:** 5-20M positions suits 256 wide. Hundreds of millions allow 512-1024.
- **Feature upgrades:** king-relative features (HalfKP or HalfKA style), output buckets by piece count.

## B.5 Training recipe

- Positions from self-play at fixed depth (6-8), skipping positions in check or mid-capture. Store as packed binary (position, score, result).
- PyTorch trainer with a sparse first layer (embedding-bag style summing), Adam, learning rate around 1e-3 with decay, batch size around 16k, validation split by game to avoid leakage.
- Loss: MSE between sigmoid of the output and the blended target, or cross-entropy on win probability.
- Data loading is the bottleneck. Use a C++ or numpy loader reading packed data, not FEN parsing in Python.
- Reference: the open-source nnue-pytorch repository. Read its sparse input handling and data loading before writing your own.
- Hardware: one consumer GPU or a free Kaggle or Colab tier.
- Deployment: export weights, quantize (int16 for the first layer, int8 for later layers), C++ inference with AVX2 and incremental accumulator updates. Verify C++ outputs match PyTorch on a few thousand positions before measuring anything.

## B.6 Phased plan

**Phase 1: Board and move generation (weeks 1-3)**
- 12 bitboards plus occupancy, make and unmake, Zobrist hashing, FEN parsing.
- Magic bitboards for sliders, lookup tables for knights, kings, pawns.
- **Perft gate:** do not continue until counts match exactly. Start position: depth 5 = 4,865,609 and depth 6 = 119,060,324. Kiwipete depth 4 = 4,085,603. Also run the other standard positions from the Chess Programming Wiki perft suite. Use perft divide against a reference engine to locate bugs. Keep as a regression suite.
- Report nodes per second.

**Phase 2: Search and handcrafted evaluation (weeks 4-6)**
- Negamax with alpha-beta, iterative deepening, quiescence search.
- Transposition table, move ordering (TT move, MVV-LVA, killers, history).
- Then null-move pruning, late move reductions, aspiration windows.
- Evaluation: material plus piece-square tables, then pawn structure and mobility, optionally Texel-tuned.
- Add features one at a time and keep only builds that gain Elo.

**Phase 3: UCI and testing infrastructure (week 7)**
- UCI commands (`position`, `go`, time management).
- Cute Chess or fastchess for matches, SPRT between your own versions.
- Optional Lichess bot for an external rating.
- Done when: a complete, resume-ready engine exists with an Elo-per-feature table.

**Phase 4: Data and first net (weeks 8-10)**
- Self-play data generator, 1-5M positions first, then scale.
- Train 768 to 256 to 32 to 1, integrate with slow, non-incremental inference first, verify against PyTorch.

**Phase 5: Make it fast (weeks 11-12)**
- Incremental accumulator updates, int16 and int8 quantization, AVX2 SIMD.
- Compare nodes per second to the handcrafted eval, then SPRT for Elo.

**Phase 6: Own experiments (weeks 13-15)**
- Bootstrap generations with an Elo curve per generation.
- One or two ablations: king-relative features, output buckets, first-layer width, lambda, training target (for example separate win/draw/loss outputs).
- Label any Stockfish use as the teacher and keep from-scratch generations separate in the results.

## B.7 Risks

- **Crowded space:** many students build chess engines. Differentiate with rigor (perft-correct, SPRT-tested) and a genuine experiment with ablations.
- **Time sink in tuning:** set a cutoff.
- **Movegen bugs poison data:** hence the perft gate.
- **Faster is not always better:** a net that is 10% better but 3x slower usually loses Elo. Always measure with matches.
- **Overlap:** an earlier chess project is commented out on the resume. Replace it, do not list both.

## B.8 Resume target

"Bitboard chess engine in C++ (X Mnps, perft-verified), UCI-compliant; trained an NNUE evaluation on N million positions, gaining +Y Elo over handcrafted evaluation (SPRT, Z games); ablations on feature set and network width."

---

## Continue in a new chat

Paste this whole file as the first message, then try:
1. "Give me a detailed checklist for Phase 1, with a perft test harness design."
2. "Design the bitboard layout and magic bitboard generation for me to implement."
3. "Plan the self-play data format and PyTorch training pipeline for Phase 4."

Open items to answer: development OS, available GPU, whether to allow Stockfish as a teacher for net v1.
