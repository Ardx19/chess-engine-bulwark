#include "tablebase.h"
extern "C" {
#include "tbprobe.h"
}

extern "C" {
uint64_t pyr_pawn_attacks(unsigned sq, unsigned c) { return PawnAttacks[c ? WHITE : BLACK][sq]; }
uint64_t pyr_knight_attacks(unsigned sq) { return KnightAttacks[sq]; }
uint64_t pyr_bishop_attacks(unsigned sq, uint64_t occ) { return bishop_attacks(sq, occ); }
uint64_t pyr_rook_attacks(unsigned sq, uint64_t occ) { return rook_attacks(sq, occ); }
uint64_t pyr_queen_attacks(unsigned sq, uint64_t occ) { return queen_attacks(sq, occ); }
uint64_t pyr_king_attacks(unsigned sq) { return KingAttacks[sq]; }
}

namespace TB {

bool init(const std::string& path) {
    Bitboards::init();
    if (path.empty() || path == "<empty>") { tb_free(); return false; }
    tb_free();
    return tb_init(path.c_str()) && TB_LARGEST > 0;
}

int largest() { return TB_LARGEST; }

int probe_wdl(const Position& pos) {
    unsigned r = tb_probe_wdl(pos.pieces(WHITE), pos.pieces(BLACK), pos.pieces(KING), pos.pieces(QUEEN),
                              pos.pieces(ROOK), pos.pieces(BISHOP), pos.pieces(KNIGHT), pos.pieces(PAWN),
                              pos.ep == SQ_NONE ? 0 : pos.ep, pos.stm == WHITE);
    return r == TB_RESULT_FAILED ? FAIL : int(r);
}

bool root_filter(const Position& pos, MoveList& allowed) {
    if (TB_LARGEST == 0 || popcount(pos.pieces()) > TB_LARGEST || pos.castling) return false;
    struct TbRootMoves rm;
    unsigned ep = pos.ep == SQ_NONE ? 0 : pos.ep;
    bool white = pos.stm == WHITE;
    int ok = tb_probe_root_dtz(pos.pieces(WHITE), pos.pieces(BLACK), pos.pieces(KING), pos.pieces(QUEEN),
                               pos.pieces(ROOK), pos.pieces(BISHOP), pos.pieces(KNIGHT), pos.pieces(PAWN),
                               pos.rule50, ep, white, false, &rm);
    if (!ok)
        ok = tb_probe_root_wdl(pos.pieces(WHITE), pos.pieces(BLACK), pos.pieces(KING), pos.pieces(QUEEN),
                               pos.pieces(ROOK), pos.pieces(BISHOP), pos.pieces(KNIGHT), pos.pieces(PAWN),
                               pos.rule50, ep, white, true, &rm);
    if (!ok || rm.size == 0) return false;

    // Pyrrhic ranks: guaranteed win == MAX, guaranteed loss == -MAX, draws (incl. cursed/blessed) in between.
    const int MAXR = 0x40000;
    int best = -MAXR - 1;
    for (unsigned i = 0; i < rm.size; ++i) best = std::max(best, rm.moves[i].tbRank);
    int lo;
    if (best == MAXR) lo = MAXR;      // won: only moves that keep the win within the 50-move limit
    else if (best >= 0) lo = 0;       // drawn: keep every move that holds the draw
    else return false;                // lost (or only blessed losses): leave the choice to search

    MoveList legal;
    generate_legal(pos, legal);
    for (unsigned i = 0; i < rm.size; ++i) {
        if (rm.moves[i].tbRank < lo) continue;
        PyrrhicMove pm = rm.moves[i].move;
        int from = PYRRHIC_MOVE_FROM(pm), to = PYRRHIC_MOVE_TO(pm);
        PieceType promo = NO_PT;
        if (PYRRHIC_MOVE_IS_QPROMO(pm)) promo = QUEEN;
        else if (PYRRHIC_MOVE_IS_RPROMO(pm)) promo = ROOK;
        else if (PYRRHIC_MOVE_IS_BPROMO(pm)) promo = BISHOP;
        else if (PYRRHIC_MOVE_IS_NPROMO(pm)) promo = KNIGHT;
        for (Move m : legal)
            if (m.from() == from && m.to() == to && (m.is_promo() ? m.promo_type() : NO_PT) == promo) {
                allowed.add(m);
                break;
            }
    }
    return allowed.size() > 0;
}

}  // namespace TB
