#pragma once
#include "types.h"
#include <bit>
#ifdef USE_PEXT
#include <immintrin.h>
#endif

constexpr Bitboard FileA = 0x0101010101010101ULL;
constexpr Bitboard FileH = FileA << 7;
constexpr Bitboard Rank1 = 0xFFULL;
constexpr Bitboard Rank2 = Rank1 << 8;
constexpr Bitboard Rank4 = Rank1 << 24;
constexpr Bitboard Rank5 = Rank1 << 32;
constexpr Bitboard Rank7 = Rank1 << 48;
constexpr Bitboard Rank8 = Rank1 << 56;

constexpr Bitboard sq_bb(int s) { return 1ULL << s; }
inline int popcount(Bitboard b) { return std::popcount(b); }
inline int lsb(Bitboard b) { return std::countr_zero(b); }
inline int pop_lsb(Bitboard& b) { int s = lsb(b); b &= b - 1; return s; }
inline bool more_than_one(Bitboard b) { return b & (b - 1); }

namespace Bitboards {
void init();   // idempotent; finds magics on first call
}

extern Bitboard PawnAttacks[COLOR_NB][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard BetweenBB[64][64];   // squares strictly between a and b if aligned, else 0
extern Bitboard LineBB[64][64];      // full line through a and b if aligned, else 0

struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard* attacks;
    int shift;
};
extern Magic RookMagics[64];
extern Magic BishopMagics[64];

inline Bitboard rook_attacks(int s, Bitboard occ) {
    const Magic& m = RookMagics[s];
#ifdef USE_PEXT
    return m.attacks[_pext_u64(occ, m.mask)];
#else
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
#endif
}
inline Bitboard bishop_attacks(int s, Bitboard occ) {
    const Magic& m = BishopMagics[s];
#ifdef USE_PEXT
    return m.attacks[_pext_u64(occ, m.mask)];
#else
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
#endif
}
inline Bitboard queen_attacks(int s, Bitboard occ) { return rook_attacks(s, occ) | bishop_attacks(s, occ); }

inline Bitboard attacks_bb(PieceType pt, int s, Bitboard occ) {
    switch (pt) {
        case KNIGHT: return KnightAttacks[s];
        case BISHOP: return bishop_attacks(s, occ);
        case ROOK:   return rook_attacks(s, occ);
        case QUEEN:  return queen_attacks(s, occ);
        case KING:   return KingAttacks[s];
        default:     return 0;
    }
}

template <Color C>
constexpr Bitboard pawn_push(Bitboard b) { return C == WHITE ? b << 8 : b >> 8; }
