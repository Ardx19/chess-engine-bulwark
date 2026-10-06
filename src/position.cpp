#include "position.h"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace Zobrist {
U64 psq[12][64];
U64 side;
U64 castling[16];
U64 ep_file[8];
void init() {
    static bool done = false;
    if (done) return;
    done = true;
    U64 s = 0x1234567890ABCDEFULL;
    auto next = [&]() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 2685821657736338717ULL; };
    for (auto& a : psq) for (auto& k : a) k = next();
    side = next();
    for (auto& k : castling) k = next();
    for (auto& k : ep_file) k = next();
}
}  // namespace Zobrist

const char* Position::START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

std::string square_name(int s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

void Position::put(Piece p, int s) {
    board[s] = p;
    by_type[type_of(p)] |= sq_bb(s);
    by_color[color_of(p)] |= sq_bb(s);
}

void Position::remove(int s) {
    Piece p = board[s];
    board[s] = NO_PIECE;
    by_type[type_of(p)] &= ~sq_bb(s);
    by_color[color_of(p)] &= ~sq_bb(s);
}

Bitboard Position::attackers_to(int s, Bitboard occ) const {
    return (PawnAttacks[BLACK][s] & pieces(WHITE, PAWN)) | (PawnAttacks[WHITE][s] & pieces(BLACK, PAWN)) |
           (KnightAttacks[s] & by_type[KNIGHT]) | (KingAttacks[s] & by_type[KING]) |
           (rook_attacks(s, occ) & (by_type[ROOK] | by_type[QUEEN])) |
           (bishop_attacks(s, occ) & (by_type[BISHOP] | by_type[QUEEN]));
}

bool Position::attacked_by(Color c, int s, Bitboard occ) const {
    return (PawnAttacks[~c][s] & pieces(c, PAWN)) || (KnightAttacks[s] & pieces(c, KNIGHT)) ||
           (KingAttacks[s] & pieces(c, KING)) ||
           (rook_attacks(s, occ) & (pieces(c, ROOK) | pieces(c, QUEEN))) ||
           (bishop_attacks(s, occ) & (pieces(c, BISHOP) | pieces(c, QUEEN)));
}

// Pieces of either color that are the only blocker between c's king and an enemy slider.
Bitboard Position::slider_blockers(Color c) const {
    int ksq = king_sq(c);
    Bitboard snipers = ((rook_attacks(ksq, 0) & (by_type[ROOK] | by_type[QUEEN])) |
                        (bishop_attacks(ksq, 0) & (by_type[BISHOP] | by_type[QUEEN]))) & pieces(~c);
    Bitboard occ = pieces(), result = 0;
    while (snipers) {
        int s = pop_lsb(snipers);
        Bitboard b = BetweenBB[ksq][s] & occ;
        if (b && !more_than_one(b)) result |= b;
    }
    return result;
}

void Position::update_state() {
    int ksq = king_sq(stm);
    checkers = attackers_to(ksq, pieces()) & pieces(~stm);
    blockers[WHITE] = slider_blockers(WHITE);
    blockers[BLACK] = slider_blockers(BLACK);
}

void Position::compute_castling_mask() {
    for (int s = 0; s < 64; ++s) castling_mask[s] = 15;
    for (int c = 0; c < 2; ++c) {
        // Only meaningful when the king is on its home rank; set when any right of that color exists.
        int bits = c == WHITE ? (WHITE_OO | WHITE_OOO) : (BLACK_OO | BLACK_OOO);
        if (!(castling & bits)) continue;
        int ksq = king_sq(Color(c));
        castling_mask[ksq] &= ~bits;
        for (int i = 0; i < 2; ++i) {
            int idx = c * 2 + i;
            if (castling & (1 << idx)) castling_mask[castling_rook[idx]] &= ~(1 << idx);
        }
    }
}

U64 Position::compute_key() const {
    U64 k = 0;
    for (int s = 0; s < 64; ++s)
        if (board[s] != NO_PIECE) k ^= Zobrist::psq[board[s]][s];
    if (stm == BLACK) k ^= Zobrist::side;
    k ^= Zobrist::castling[castling];
    if (ep != SQ_NONE) k ^= Zobrist::ep_file[file_of(ep)];
    return k;
}

bool Position::set_fen(const std::string& fen) {
    Bitboards::init();
    Zobrist::init();
    for (auto& b : by_type) b = 0;
    for (auto& b : by_color) b = 0;
    for (auto& p : board) p = NO_PIECE;
    for (auto& r : castling_rook) r = SQ_NONE;
    castling = 0;
    ep = SQ_NONE;
    rule50 = 0;
    fullmove = 1;

    std::istringstream ss(fen);
    std::string placement, side, cast, epstr;
    if (!(ss >> placement >> side)) return false;
    if (!(ss >> cast)) cast = "-";
    if (!(ss >> epstr)) epstr = "-";
    int hm = 0, fm = 1;
    ss >> hm >> fm;
    rule50 = hm;
    fullmove = std::max(fm, 1);

    int f = 0, r = 7;
    const std::string names = "PNBRQKpnbrqk";
    for (char ch : placement) {
        if (ch == '/') { if (f != 8) return false; f = 0; --r; if (r < 0) return false; }
        else if (ch >= '1' && ch <= '8') f += ch - '0';
        else {
            size_t i = names.find(ch);
            if (i == std::string::npos || f > 7) return false;
            Piece p = i < 6 ? make_piece(WHITE, PieceType(i)) : make_piece(BLACK, PieceType(i - 6));
            put(p, make_sq(f++, r));
        }
        if (f > 8) return false;
    }
    if (r != 0 || f != 8) return false;
    if (popcount(pieces(WHITE, KING)) != 1 || popcount(pieces(BLACK, KING)) != 1) return false;

    if (side == "w") stm = WHITE; else if (side == "b") stm = BLACK; else return false;

    for (char ch : cast) {
        if (ch == '-') break;
        Color c = std::isupper((unsigned char)ch) ? WHITE : BLACK;
        int back = c == WHITE ? 0 : 7;
        Bitboard rooks = pieces(c, ROOK) & (Rank1 << (8 * back));
        int ksq = king_sq(c);
        if (rank_of(ksq) != back) return false;
        int rsq = SQ_NONE;
        char u = (char)std::toupper((unsigned char)ch);
        bool king_side;
        if (u == 'K' || u == 'Q') {
            king_side = u == 'K';
            // X-FEN: outermost rook on that side of the king.
            if (king_side) { for (int s = make_sq(7, back); s > ksq; --s) if (rooks & sq_bb(s)) { rsq = s; break; } }
            else           { for (int s = make_sq(0, back); s < ksq; ++s) if (rooks & sq_bb(s)) { rsq = s; break; } }
        } else if (u >= 'A' && u <= 'H') {
            rsq = make_sq(u - 'A', back);
            if (!(rooks & sq_bb(rsq))) return false;
            king_side = rsq > ksq;
        } else return false;
        if (rsq == SQ_NONE) return false;
        int idx = c * 2 + (king_side ? 0 : 1);
        castling |= 1 << idx;
        castling_rook[idx] = rsq;
    }

    if (epstr != "-" && epstr.size() == 2) {
        int s = make_sq(epstr[0] - 'a', epstr[1] - '1');
        if (s >= 0 && s < 64) {
            // Keep only if a pseudo-legal en passant capture exists (so the hash is consistent).
            Color us = stm;
            if (PawnAttacks[~us][s] & pieces(us, PAWN)) ep = s;
        }
    }
    compute_castling_mask();
    key = compute_key();
    update_state();
    return true;
}

std::string Position::fen(bool shredder) const {
    std::string out;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = board[make_sq(f, r)];
            if (p == NO_PIECE) { ++empty; continue; }
            if (empty) { out += char('0' + empty); empty = 0; }
            out += "PNBRQKpnbrqk"[p];
        }
        if (empty) out += char('0' + empty);
        if (r) out += '/';
    }
    out += stm == WHITE ? " w " : " b ";
    if (!castling) out += '-';
    else {
        for (int idx : {0, 1, 2, 3}) {
            if (!(castling & (1 << idx))) continue;
            char ch;
            if (shredder) ch = char('a' + file_of(castling_rook[idx]));
            else ch = (idx & 1) ? 'q' : 'k';
            out += (idx < 2) ? (char)std::toupper(ch) : ch;
        }
    }
    out += ' ';
    out += ep == SQ_NONE ? "-" : square_name(ep);
    out += ' ' + std::to_string(rule50) + ' ' + std::to_string(fullmove);
    return out;
}

bool Position::is_legal(Move m) const {
    Color us = stm;
    int from = m.from(), to = m.to();
    int ksq = king_sq(us);

    if (m.is_castle()) return true;   // fully verified by the generator

    if (m.is_ep()) {
        int cap = to + (us == WHITE ? -8 : 8);
        Bitboard occ = (pieces() ^ sq_bb(from) ^ sq_bb(cap)) | sq_bb(to);
        Bitboard enemy = pieces(~us) & ~sq_bb(cap);
        return !((rook_attacks(ksq, occ) & (enemy & (by_type[ROOK] | by_type[QUEEN]))) ||
                 (bishop_attacks(ksq, occ) & (enemy & (by_type[BISHOP] | by_type[QUEEN]))) ||
                 (KnightAttacks[ksq] & (enemy & by_type[KNIGHT])) ||
                 (PawnAttacks[us][ksq] & (enemy & by_type[PAWN])));
    }

    if (type_of(board[from]) == KING) {
        Bitboard occ = pieces() ^ sq_bb(from);
        return !attacked_by(~us, to, occ & ~sq_bb(to)) ;
    }

    if ((blockers[us] & sq_bb(from)) && !(LineBB[from][to] & sq_bb(ksq))) return false;
    return true;
}

Position Position::make(Move m) const {
    Position p = *this;
    Color us = stm;
    int from = m.from(), to = m.to();
    Piece pc = board[from];

    if (p.ep != SQ_NONE) p.key ^= Zobrist::ep_file[file_of(p.ep)];
    p.ep = SQ_NONE;
    p.key ^= Zobrist::castling[p.castling];
    ++p.rule50;

    if (m.is_castle()) {
        int idx = us * 2 + (to > from ? 0 : 1);
        int rook_from = to;
        bool king_side = to > from;
        int back = us == WHITE ? 0 : 56;
        int kto = back + (king_side ? G1 : C1), rto = back + (king_side ? F1 : D1);
        (void)idx;
        Piece rk = make_piece(us, ROOK);
        p.remove(from);
        p.remove(rook_from);
        p.put(pc, kto);
        p.put(rk, rto);
        p.key ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][kto] ^ Zobrist::psq[rk][rook_from] ^ Zobrist::psq[rk][rto];
    } else {
        if (m.is_capture()) {
            int cap_sq = m.is_ep() ? to + (us == WHITE ? -8 : 8) : to;
            Piece cp = board[cap_sq];
            p.remove(cap_sq);
            p.key ^= Zobrist::psq[cp][cap_sq];
            p.rule50 = 0;
        }
        p.remove(from);
        p.key ^= Zobrist::psq[pc][from];
        Piece np = m.is_promo() ? make_piece(us, m.promo_type()) : pc;
        p.put(np, to);
        p.key ^= Zobrist::psq[np][to];
        if (type_of(pc) == PAWN) {
            p.rule50 = 0;
            if (m.flags() == F_DOUBLE) {
                int eps = (from + to) / 2;
                if (PawnAttacks[us][eps] & p.pieces(~us, PAWN)) {
                    p.ep = eps;
                    p.key ^= Zobrist::ep_file[file_of(eps)];
                }
            }
        }
    }

    p.castling &= p.castling_mask[from] & p.castling_mask[to];
    p.key ^= Zobrist::castling[p.castling];
    p.stm = ~us;
    p.key ^= Zobrist::side;
    if (us == BLACK) ++p.fullmove;
    p.update_state();
    return p;
}

Position Position::make_null() const {
    Position p = *this;
    if (p.ep != SQ_NONE) p.key ^= Zobrist::ep_file[file_of(p.ep)];
    p.ep = SQ_NONE;
    p.stm = ~stm;
    p.key ^= Zobrist::side;
    ++p.rule50;
    p.update_state();
    return p;
}

bool Position::insufficient_material() const {
    if (by_type[PAWN] || by_type[ROOK] || by_type[QUEEN]) return false;
    int minors = popcount(by_type[KNIGHT] | by_type[BISHOP]);
    if (minors <= 1) return true;
    // Only bishops, all on one square color.
    if (!by_type[KNIGHT]) {
        constexpr Bitboard light = 0x55AA55AA55AA55AAULL;
        Bitboard b = by_type[BISHOP];
        return !(b & light) || !(b & ~light);
    }
    return false;
}

std::string Position::move_to_uci(Move m, bool chess960) const {
    if (!m) return "0000";
    int from = m.from(), to = m.to();
    if (m.is_castle() && !chess960) {
        bool ks = to > from;
        int back = rank_of(from) * 8;
        to = back + (ks ? G1 : C1);
    }
    std::string s = square_name(from) + square_name(to);
    if (m.is_promo()) s += "nbrq"[m.promo_type() - KNIGHT];
    return s;
}
