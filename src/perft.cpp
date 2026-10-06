#include "perft.h"
#include "movegen.h"
#include <cstdio>

U64 perft(const Position& pos, int depth) {
    MoveList ml;
    generate_legal(pos, ml);
    if (depth <= 1) return ml.size();
    U64 n = 0;
    for (Move m : ml) n += perft(pos.make(m), depth - 1);
    return n;
}

void perft_divide(const Position& pos, int depth, bool chess960) {
    MoveList ml;
    generate_legal(pos, ml);
    U64 total = 0;
    for (Move m : ml) {
        U64 n = depth <= 1 ? 1 : perft(pos.make(m), depth - 1);
        std::printf("%s: %llu\n", pos.move_to_uci(m, chess960).c_str(), (unsigned long long)n);
        total += n;
    }
    std::printf("\nNodes: %llu\n", (unsigned long long)total);
}
