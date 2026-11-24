# Frozen interfaces (Day 0)

Changing anything here means updating every workstream. Source of truth for B.3 / B.7 / B.9 of
`project_B_nnue_chess_engine.md`.

## Board / move encoding (implemented: `src/types.h`)
- LERF squares (a1 = 0, h8 = 63). Piece = color*6 + type (P N B R Q K).
- Move = 16 bit: from | to << 6 | flags << 12. Castling = king-takes-own-rook. Flags: 0 quiet, 1 double push,
  2 castle, 4 capture, 5 ep, 8-11 promo N/B/R/Q, 12-15 promo-capture N/B/R/Q.
- En passant square is kept in `Position` only if a pseudo-legal ep capture exists (so FEN output omits it otherwise).

## Master record (56 bytes, little endian, versioned shard header)
| Offset | Size | Field |
|---|---|---|
| 0 | 8 | occupancy bitboard (LERF) |
| 8 | 16 | 4-bit piece codes for occupied squares in ascending square order (code = Piece 0-11) |
| 24 | 1 | stm (bit 0) / castling rights (bits 1-4) |
| 25 | 1 | ep file+1 (0 = none) |
| 26 | 1 | rule50 (saturating at 255) |
| 27 | 2 | sf_cp int16 (sentinel -32768 = absent) |
| 29 | 4 | sf_w, sf_d per-mille uint16 (0xFFFF = absent); L = 1000 - W - D |
| 33 | 2 | lc0_q int16 (q * 10000) |
| 35 | 2 | lc0_d uint16 (d * 10000) |
| 37 | 2 | own_cp int16 (sentinel -32768 = absent) |
| 39 | 1 | result (0 loss, 1 draw, 2 win, from stm view; 255 unknown) |
| 40 | 1 | flags: bit0 tablebase-overridden, bits 1-7 source id |
| 41 | 2 | ply in game |
| 43 | 8 | game id = session << 32 | counter |
| 51 | 5 | reserved (zero) |
Total 56. Shard header: magic "NNUEMR01", version u32, record count u64, shard id u32.

## Training record (40 bytes)
Master record without provenance (ply, game id, own_cp, reserved): board 27 B + teacher fields + result/flags, padded to 40.

## UCI option / switch names (all defense features are switches)
`UCI_Chess960`, `Hash`, `Threads`, `Move Overhead`, `EvalFile`, `SyzygyPath`, `UCI_ShowWDL`,
`DrawValue` (spin, per-mille, default 500), `DefenseMode` (check; D2), `DefenseDrawValue` (spin, 650),
`DefenseLossThreshold` (spin, per-mille P(loss), 400), `D1_WDLUtility`, `D3_Rule50Scale`,
`D4_Syzygy`, `D5_StalemateGuard`, `D6_PerpetualSeek`, `D7_FortressDetect`, plus the ladder switches
(`Aspiration`, `RFP`, `NMP`, `LMR`, `KillersHistory`, `LMPFutility`, `IIR`, `Improving`, `SEEPrune`, `TMScaling`).
Draw weight: `d = 0.5` unless `D1_WDLUtility` (then `DrawValue`) or D2 triggers (then `DefenseDrawValue`);
opponent-to-move nodes use `1 - d`; in-tree draws score `400*logit(d)` for the root side. The TT is cleared when
`d`, the root side (if `d != 0.5`) or the net changes.
Extra commands: `bench [depth] [net] [verbose]`, `verifynet <net> <ref.jsonl> [limit]`, `perft <depth>`, `eval`, `d`,
`go ... searchmoves <moves>`.

## Net file format
`.nnue` (quantised, magic "NNUEWDL1") and `.nnuef` (float32, magic "NNUEFLT1"), little endian, sha256 trailer.
Scale: QA = 255, QB = 64. Full layouts in the docstring of `nnue/export.py`.
- Version 1 (N1 = arch 1, N1-WDL = arch 2): u32 version, arch, L1, buckets = 1, heads | i16 ft_w[768][L1] | i16 ft_b |
  i16 out_w[heads][2*L1] | i32 out_b[heads].
- Version 2 (N2 = arch 3, N3 = arch 4): u32 version, arch, L1, buckets = 8, heads, L2, L3, K, router (0 none, 1 piece
  count, 2 learned) | i16 ft_w[8*768][L1] | i16 ft_b | i8 l2_w[K*L2][2*L1] | f32 l2_b, l3_w, l3_b, out_w, out_b |
  (router 2: i8 r_w[K][2*L1], f32 r_s[K][4], f32 r_b[K]).
- N2/N3 features: own king seen from the perspective; files e-h mirror every square (^7); bucket by
  `[0 1 2 3 / 4 4 5 5 / 6 6 6 6 / 6 6 6 6 / 7 x16]` (rank-major over the mirrored a-d files);
  feature = bucket*768 + (rel*6 + type)*64 + sq.

## Manifest schema (`manifest.json`)
`{ "schema":1, "lc0_run":..., "files":[...], "filter":{...,"kept":n,"seen":n}, "stockfish":{"version","sha256","nodes","hash","threads","tb"},
"shards":[{"name","records","sha256"}] }`
