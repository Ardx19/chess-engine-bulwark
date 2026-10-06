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
`UCI_Chess960`, `Hash`, `Threads`, `Move Overhead`, `EvalFile`, `SyzygyPath`,
`DrawValue` (spin, per-mille, default 500), `DefenseMode` (check), `D1_WDLUtility`, `D3_Rule50Scale`,
`D4_Syzygy`, `D5_StalemateGuard`, `D6_PerpetualSeek`, `D7_FortressDetect`.
Extra commands: `bench [depth]`, `verifynet <file.jsonl>`, `perft <depth>`.

## Net file format
`.nnue`: magic "NNUEWDL1", u32 version, u32 arch id (N1=1, N1-WDL=2, N2=3, N3=4), u32 L1 width, u32 bucket count,
u32 head count, then little-endian int16 feature weights/biases, int8 hidden weights, int32 biases, sha256 trailer.
Scale: QA = 255, QB = 64.

## Manifest schema (`manifest.json`)
`{ "schema":1, "lc0_run":..., "files":[...], "filter":{...,"kept":n,"seen":n}, "stockfish":{"version","sha256","nodes","hash","threads","tb"},
"shards":[{"name","records","sha256"}] }`
