#include "bitboard.h"
#include <cstdlib>
#include <vector>

Bitboard PawnAttacks[COLOR_NB][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];
Magic RookMagics[64];
Magic BishopMagics[64];

namespace {

Bitboard RookTable[0x19000];   // fancy magic: 4 * 2^12 worst case summed ~ 102400 entries
Bitboard BishopTable[0x1480];

bool on_board(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }

Bitboard slide(int s, Bitboard occ, const int (*dirs)[2]) {
    Bitboard att = 0;
    for (int d = 0; d < 4; ++d) {
        int f = file_of(s) + dirs[d][0], r = rank_of(s) + dirs[d][1];
        while (on_board(f, r)) {
            Bitboard b = sq_bb(make_sq(f, r));
            att |= b;
            if (occ & b) break;
            f += dirs[d][0];
            r += dirs[d][1];
        }
    }
    return att;
}

const int RookDirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
const int BishopDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

struct Rng {
    U64 s = 0x9E3779B97F4A7C15ULL;
    U64 next() {   // xorshift64*
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    U64 sparse() { return next() & next() & next(); }
};

void init_magics(Magic* magics, Bitboard* table, const int (*dirs)[2], Rng& rng) {
    Bitboard* next_slot = table;
    for (int s = 0; s < 64; ++s) {
        // Relevant occupancy mask: ray squares excluding board edges (unless the piece is on that edge's line).
        Bitboard edges = ((Rank1 | Rank8) & ~(Rank1 << (8 * rank_of(s)))) | ((FileA | FileH) & ~(FileA << file_of(s)));
        Magic& m = magics[s];
        m.mask = slide(s, 0, dirs) & ~edges;
        int bits = popcount(m.mask);
        m.shift = 64 - bits;
        m.attacks = next_slot;
        size_t size = size_t(1) << bits;
        next_slot += size;

        std::vector<Bitboard> occ(size), ref(size);
        Bitboard b = 0;
        for (size_t i = 0; i < size; ++i) {   // carry-rippler subset enumeration
            occ[i] = b;
            ref[i] = slide(s, b, dirs);
            b = (b - m.mask) & m.mask;
        }
        std::vector<int> epoch(size, 0);
        int attempt = 0;
        for (;;) {
            m.magic = rng.sparse();
            if (popcount((m.mask * m.magic) >> 56) < 6) continue;
            ++attempt;
            bool ok = true;
            for (size_t i = 0; i < size && ok; ++i) {
                size_t idx = (occ[i] * m.magic) >> m.shift;
                if (epoch[idx] != attempt) { epoch[idx] = attempt; m.attacks[idx] = ref[i]; }
                else if (m.attacks[idx] != ref[i]) ok = false;
            }
            if (ok) break;
        }
    }
}

}  // namespace

void Bitboards::init() {
    static bool done = false;
    if (done) return;
    done = true;

    Rng rng;
    init_magics(RookMagics, RookTable, RookDirs, rng);
    init_magics(BishopMagics, BishopTable, BishopDirs, rng);

    const int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    for (int s = 0; s < 64; ++s) {
        int f = file_of(s), r = rank_of(s);
        Bitboard n = 0, k = 0;
        for (auto& d : kn) if (on_board(f + d[0], r + d[1])) n |= sq_bb(make_sq(f + d[0], r + d[1]));
        for (int df = -1; df <= 1; ++df)
            for (int dr = -1; dr <= 1; ++dr)
                if ((df || dr) && on_board(f + df, r + dr)) k |= sq_bb(make_sq(f + df, r + dr));
        KnightAttacks[s] = n;
        KingAttacks[s] = k;
        Bitboard p = sq_bb(s);
        PawnAttacks[WHITE][s] = ((p & ~FileA) << 7) | ((p & ~FileH) << 9);
        PawnAttacks[BLACK][s] = ((p & ~FileH) >> 7) | ((p & ~FileA) >> 9);
    }

    for (int a = 0; a < 64; ++a)
        for (int b = 0; b < 64; ++b) {
            BetweenBB[a][b] = LineBB[a][b] = 0;
            if (a == b) continue;
            if (rook_attacks(a, 0) & sq_bb(b)) {
                LineBB[a][b] = (rook_attacks(a, 0) & rook_attacks(b, 0)) | sq_bb(a) | sq_bb(b);
                BetweenBB[a][b] = rook_attacks(a, sq_bb(b)) & rook_attacks(b, sq_bb(a));
            } else if (bishop_attacks(a, 0) & sq_bb(b)) {
                LineBB[a][b] = (bishop_attacks(a, 0) & bishop_attacks(b, 0)) | sq_bb(a) | sq_bb(b);
                BetweenBB[a][b] = bishop_attacks(a, sq_bb(b)) & bishop_attacks(b, sq_bb(a));
            }
        }
}
