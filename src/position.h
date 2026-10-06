#pragma once
#include "bitboard.h"
#include <string>

namespace Zobrist {
extern U64 psq[12][64];
extern U64 side;
extern U64 castling[16];
extern U64 ep_file[8];
void init();
}

// Copy-make position: make() returns a new Position, so the search keeps a ply-indexed stack of them.
struct Position {
    Bitboard by_type[PT_NB];
    Bitboard by_color[COLOR_NB];
    Piece board[64];
    Color stm;
    int castling;              // CastlingRight bitmask
    int ep;                    // en passant target square, or SQ_NONE. Only set if a pseudo-legal capture exists.
    int rule50;
    int fullmove;
    int castling_rook[4];      // rook origin square per castling right bit index (WK, WQ, BK, BQ)
    int castling_mask[64];     // rights that survive a move touching this square
    U64 key;
    Bitboard checkers;         // enemy pieces giving check to the side to move
    Bitboard blockers[COLOR_NB];  // pieces (either color) that shield the given color's king from a slider

    static const char* START_FEN;

    bool set_fen(const std::string& fen);     // standard FEN, X-FEN or Shredder-FEN; false on parse failure
    std::string fen(bool shredder = false) const;

    Bitboard pieces() const { return by_color[WHITE] | by_color[BLACK]; }
    Bitboard pieces(PieceType pt) const { return by_type[pt]; }
    Bitboard pieces(Color c) const { return by_color[c]; }
    Bitboard pieces(Color c, PieceType pt) const { return by_color[c] & by_type[pt]; }
    Piece piece_on(int s) const { return board[s]; }
    int king_sq(Color c) const { return lsb(pieces(c, KING)); }
    bool in_check() const { return checkers != 0; }

    Bitboard attackers_to(int s, Bitboard occ) const;       // pieces of both colors
    bool attacked_by(Color c, int s, Bitboard occ) const;   // is s attacked by color c given occupancy occ
    bool is_legal(Move m) const;                            // for moves from the pseudo-legal generator
    Position make(Move m) const;
    Position make_null() const;
    U64 compute_key() const;
    bool has_non_pawn_material(Color c) const { return pieces(c) & ~pieces(c, PAWN) & ~pieces(c, KING); }
    bool insufficient_material() const;

    // Parse a UCI move ("e2e4", "e7e8q", castling as king-takes-rook "e1h1" or classical "e1g1") for this position.
    Move parse_move(const std::string& s) const;
    std::string move_to_uci(Move m, bool chess960) const;

    void put(Piece p, int s);
    void remove(int s);
    void update_state();    // checkers, blockers
    void compute_castling_mask();
    Bitboard slider_blockers(Color c) const;
};

std::string square_name(int s);
