#pragma once
#include <cstdint>

using U64 = uint64_t;
using Bitboard = uint64_t;

// LERF mapping: a1 = 0 ... h8 = 63, north = +8.
enum Square : int { A1, B1, C1, D1, E1, F1, G1, H1, A2 = 8, A8 = 56, H8 = 63, SQ_NONE = 64 };
enum Color : int { WHITE, BLACK, COLOR_NB = 2 };
enum PieceType : int { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PT_NB = 6, NO_PT = 6 };
// Piece = color * 6 + type
enum Piece : int { W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                   B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING, NO_PIECE = 12 };

constexpr Color operator~(Color c) { return Color(c ^ 1); }
constexpr Piece make_piece(Color c, PieceType pt) { return Piece(c * 6 + pt); }
constexpr PieceType type_of(Piece p) { return PieceType(p % 6); }
constexpr Color color_of(Piece p) { return Color(p / 6); }
constexpr int file_of(int s) { return s & 7; }
constexpr int rank_of(int s) { return s >> 3; }
constexpr int make_sq(int file, int rank) { return rank * 8 + file; }
constexpr int relative_rank(Color c, int s) { return c == WHITE ? rank_of(s) : 7 - rank_of(s); }
constexpr int relative_sq(Color c, int s) { return c == WHITE ? s : s ^ 56; }

// Castling rights bits; index into rook-square arrays is the bit position.
enum CastlingRight : int { WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8 };

// 16-bit move: from(6) | to(6) << 6 | flags(4) << 12.
// Castling is encoded as king-captures-own-rook (from = king square, to = rook square).
enum MoveFlag : int {
    F_QUIET = 0, F_DOUBLE = 1, F_CASTLE = 2,
    F_CAPTURE = 4, F_EP = 5,
    F_PROMO = 8,        // + (pt - KNIGHT): N=8, B=9, R=10, Q=11
    F_PROMO_CAP = 12    // + (pt - KNIGHT)
};

struct Move {
    uint16_t v = 0;
    constexpr Move() = default;
    constexpr explicit Move(uint16_t x) : v(x) {}
    constexpr Move(int from, int to, int flags) : v(uint16_t(from | (to << 6) | (flags << 12))) {}
    constexpr int from() const { return v & 63; }
    constexpr int to() const { return (v >> 6) & 63; }
    constexpr int flags() const { return v >> 12; }
    constexpr bool is_capture() const { return (flags() & 4) != 0; }       // includes ep and promo-capture
    constexpr bool is_promo() const { return (flags() & 8) != 0; }
    constexpr bool is_ep() const { return flags() == F_EP; }
    constexpr bool is_castle() const { return flags() == F_CASTLE; }
    constexpr PieceType promo_type() const { return PieceType(KNIGHT + (flags() & 3)); }
    constexpr bool operator==(Move o) const { return v == o.v; }
    constexpr bool operator!=(Move o) const { return v != o.v; }
    constexpr explicit operator bool() const { return v != 0; }
};
constexpr Move MOVE_NONE{};
