#include "movegen.h"

namespace {

template <Color Us, GenType T>
void gen_pawns(const Position& pos, MoveList& list, Bitboard target) {
    constexpr Color Them = ~Us;
    constexpr int Up = Us == WHITE ? 8 : -8;
    constexpr int UpLeft = Us == WHITE ? 7 : -9;
    constexpr int UpRight = Us == WHITE ? 9 : -7;
    constexpr Bitboard Rank3 = Us == WHITE ? Rank1 << 16 : Rank1 << 40;
    constexpr bool Caps = T != GEN_QUIETS, Quiets = T != GEN_CAPTURES;

    Bitboard pawns = pos.pieces(Us, PAWN);
    Bitboard empty = ~pos.pieces();
    Bitboard enemies = pos.pieces(Them);
    Bitboard promo_from = pawns & (Us == WHITE ? Rank7 : Rank2);
    Bitboard others = pawns & ~promo_from;

    auto add_promos = [&](int from, int to, bool cap) {
        int base = cap ? F_PROMO_CAP : F_PROMO;
        for (int i = 3; i >= 0; --i) list.add(Move(from, to, base + i));
    };

    if (Quiets) {
        Bitboard b1 = pawn_push<Us>(others) & empty;
        Bitboard b2 = pawn_push<Us>(b1 & Rank3) & empty;
        b1 &= target; b2 &= target;
        while (b1) { int to = pop_lsb(b1); list.add(Move(to - Up, to, F_QUIET)); }
        while (b2) { int to = pop_lsb(b2); list.add(Move(to - 2 * Up, to, F_DOUBLE)); }
    }
    if (Caps) {
        Bitboard b1 = pawn_push<Us>(promo_from) & empty & target;   // promotion pushes count as "captures" bucket
        while (b1) { int to = pop_lsb(b1); add_promos(to - Up, to, false); }
    }
    if (Caps) {
        Bitboard l = (Us == WHITE ? (others & ~FileA) << 7 : (others & ~FileA) >> 9) & enemies & target;
        Bitboard r = (Us == WHITE ? (others & ~FileH) << 9 : (others & ~FileH) >> 7) & enemies & target;
        while (l) { int to = pop_lsb(l); list.add(Move(to - UpLeft, to, F_CAPTURE)); }
        while (r) { int to = pop_lsb(r); list.add(Move(to - UpRight, to, F_CAPTURE)); }
        Bitboard pl = (Us == WHITE ? (promo_from & ~FileA) << 7 : (promo_from & ~FileA) >> 9) & enemies & target;
        Bitboard pr = (Us == WHITE ? (promo_from & ~FileH) << 9 : (promo_from & ~FileH) >> 7) & enemies & target;
        while (pl) { int to = pop_lsb(pl); add_promos(to - UpLeft, to, true); }
        while (pr) { int to = pop_lsb(pr); add_promos(to - UpRight, to, true); }

        if (pos.ep != SQ_NONE) {
            int capsq = pos.ep - Up;
            // In check, the capture must remove the checker or block; otherwise `target` is everything.
            if ((target & (sq_bb(pos.ep) | sq_bb(capsq)))) {
                Bitboard from = PawnAttacks[Them][pos.ep] & pawns;
                while (from) list.add(Move(pop_lsb(from), pos.ep, F_EP));
            }
        }
    }
}

template <Color Us>
void gen_castling(const Position& pos, MoveList& list) {
    constexpr int back = Us == WHITE ? 0 : 56;
    int ksq = pos.king_sq(Us);
    Bitboard occ = pos.pieces();
    for (int side = 0; side < 2; ++side) {
        int idx = Us * 2 + side;
        if (!(pos.castling & (1 << idx))) continue;
        int rsq = pos.castling_rook[idx];
        int kto = back + (side == 0 ? G1 : C1), rto = back + (side == 0 ? F1 : D1);
        auto span = [](int a, int b) {   // inclusive range of squares on one rank
            int lo = a < b ? a : b, hi = a < b ? b : a;
            return ((2ULL << hi) - 1) & ~((1ULL << lo) - 1);
        };
        Bitboard path = span(ksq, kto) | span(rsq, rto);
        Bitboard occ_wo = occ & ~sq_bb(ksq) & ~sq_bb(rsq);
        if (path & occ_wo) continue;
        // King may not pass through or land on attacked squares. Slider attacks are computed with the king and
        // the castling rook removed from occupancy (a rook shielding the king would otherwise hide an attacker).
        Bitboard ksp = (span(ksq, kto) & ~sq_bb(ksq)) | sq_bb(kto);   // start square is covered by the not-in-check
        // precondition, but the destination must always be tested (it may equal the start square)
        bool ok = true;
        while (ksp && ok) {
            int s = pop_lsb(ksp);
            if (pos.attacked_by(~Us, s, occ_wo)) ok = false;
        }
        if (ok) list.add(Move(ksq, rsq, F_CASTLE));
    }
}

template <Color Us, GenType T>
void gen_all(const Position& pos, MoveList& list) {
    constexpr bool Caps = T != GEN_QUIETS, Quiets = T != GEN_CAPTURES;
    int ksq = pos.king_sq(Us);
    Bitboard occ = pos.pieces();
    Bitboard enemies = pos.pieces(~Us);
    Bitboard target_all = ~0ULL;   // pawn target (pawn code applies capture/push masks itself)
    Bitboard piece_target;

    bool evasion = pos.in_check();
    if (evasion && more_than_one(pos.checkers)) {
        // Double check: king moves only.
    } else {
        if (evasion) {
            int csq = lsb(pos.checkers);
            target_all = BetweenBB[ksq][csq] | sq_bb(csq);
        }
        piece_target = (Caps ? enemies : 0) | (Quiets ? ~occ : 0);
        piece_target &= target_all & ~pos.pieces(Us);
        gen_pawns<Us, T>(pos, list, evasion ? target_all : ~0ULL);
        for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
            Bitboard pcs = pos.pieces(Us, PieceType(pt));
            while (pcs) {
                int from = pop_lsb(pcs);
                Bitboard att = attacks_bb(PieceType(pt), from, occ) & piece_target;
                while (att) {
                    int to = pop_lsb(att);
                    list.add(Move(from, to, (pos.board[to] != NO_PIECE) ? F_CAPTURE : F_QUIET));
                }
            }
        }
    }
    Bitboard kt = ((Caps ? enemies : 0) | (Quiets ? ~occ : 0)) & ~pos.pieces(Us);
    Bitboard katt = KingAttacks[ksq] & kt;
    while (katt) {
        int to = pop_lsb(katt);
        list.add(Move(ksq, to, (pos.board[to] != NO_PIECE) ? F_CAPTURE : F_QUIET));
    }
    if (Quiets && !evasion) gen_castling<Us>(pos, list);
}

}  // namespace

template <GenType T>
void generate(const Position& pos, MoveList& list) {
    if (pos.stm == WHITE) gen_all<WHITE, T>(pos, list);
    else gen_all<BLACK, T>(pos, list);
}
template void generate<GEN_CAPTURES>(const Position&, MoveList&);
template void generate<GEN_QUIETS>(const Position&, MoveList&);
template void generate<GEN_ALL>(const Position&, MoveList&);

void generate_legal(const Position& pos, MoveList& list) {
    MoveList tmp;
    generate<GEN_ALL>(pos, tmp);
    for (Move m : tmp)
        if (pos.is_legal(m)) list.add(m);
}

Move Position::parse_move(const std::string& s) const {
    MoveList ml;
    generate_legal(*this, ml);
    // Plain moves first: a king step "f1g1" can also be the classical spelling of a castle.
    for (Move m : ml)
        if (!m.is_castle() && move_to_uci(m, true) == s) return m;
    for (Move m : ml)
        if (m.is_castle() && (move_to_uci(m, true) == s || move_to_uci(m, false) == s)) return m;
    return MOVE_NONE;
}
