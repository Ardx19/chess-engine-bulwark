#pragma once
#include "position.h"

struct MoveList {
    Move moves[256];
    int count = 0;
    void add(Move m) { moves[count++] = m; }
    Move* begin() { return moves; }
    Move* end() { return moves + count; }
    const Move* begin() const { return moves; }
    const Move* end() const { return moves + count; }
    int size() const { return count; }
};

enum GenType { GEN_CAPTURES, GEN_QUIETS, GEN_ALL };
// CAPTURES: captures, en passant and all promotions. QUIETS: the rest (incl. castling). When in check both
// produce evasions only, split the same way. Output is pseudo-legal: filter with Position::is_legal.
template <GenType T> void generate(const Position& pos, MoveList& list);

void generate_legal(const Position& pos, MoveList& list);
