#include "bench.h"
#include <chrono>
#include <cstdio>

static const char* BenchFens[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9",
    "rn1qkb1r/pp2pppp/2p2n2/5b2/2pP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6",
    "r2qkb1r/pp1n1ppp/2p1pn2/3p1b2/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 7",
    "2rq1rk1/pp2bppp/2n1pn2/3p4/3P1B2/2NBPN2/PP3PPP/2RQ1RK1 w - - 0 11",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "8/8/4k3/8/2p5/2P1K3/8/8 w - - 0 1",
    "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1",
    "8/8/8/8/5k2/8/5P2/5K2 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "8/6pp/5k2/8/5K2/8/6PP/8 w - - 0 1",
    "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
    "8/8/8/4k3/8/8/2B1K3/5N2 w - - 0 1",
    "4r1k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "r1b2rk1/2q1bppp/p2p1n2/np2p3/3NP3/1BN1BP2/PPP3PP/R2QK2R w KQ - 0 11",
    "r4rk1/ppp2ppp/2n1bn2/3qp3/8/2PP1N2/PP1NBPPP/R2Q1RK1 w - - 0 10",
    "2r2rk1/1p1nqppp/p2bpn2/3p4/3P1B2/2NBPN1P/PP2QPP1/2R2RK1 w - - 0 15",
    "r3r1k1/pp3ppp/2p2n2/q7/3P4/2P1B3/PP1Q1PPP/R3R1K1 w - - 0 16",
    "8/pp3k2/2p2p2/2P1p1p1/P3P3/1P3KP1/5P2/8 w - - 0 40",
    "3r2k1/pp3pp1/2p1b2p/8/2P5/1P2B1P1/P4P1P/3R2K1 w - - 0 25",
    "5rk1/pp4pp/2p5/2P1p3/1P2P3/P4P2/6PP/R5K1 b - - 0 30",
    "2k5/8/1K6/8/8/8/8/R7 w - - 0 1",
    "8/3k4/8/3KP3/8/8/8/8 w - - 0 1",
    "1k6/1P6/1K6/8/8/8/8/8 b - - 0 1",
    "r1bqkb1r/pppp1Bpp/2n2n2/4p3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 0 4",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8",
    "rnb1kbnr/pp3ppp/2p1p3/q2pP3/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 1 6",
    "r1bq1rk1/ppp1nppp/3p1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7",
    "2kr3r/ppp1qppp/2n1b3/3pP3/3P4/2P2N2/PP2QPPP/R1B1K2R w KQ - 0 12",
    "r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 5 6",
    "6k1/6pp/8/8/8/8/1q4PP/3Q2K1 w - - 0 1",
    "5k2/5p2/5P2/8/3K4/8/8/8 w - - 0 1",
    "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
    "bnnrqbkr/ppp1ppp1/8/P2p3p/8/5N2/1PPPPPPP/BRKR1QNB b DBhd - 0 3",
    "nqk1brnb/ppr1pppp/8/2pp4/1PP5/5P1P/P2PP1P1/NBQRNKBR b HD - 0 5",
    "rkr3nq/ppp1pppp/8/8/8/8/PPP1PPPP/RKR3NQ w CAca - 0 1",
    "1r1k1r2/8/8/8/8/8/8/1R1K1R2 w FBfb - 0 1",
};

std::vector<const char*> bench_fens() { return {std::begin(BenchFens), std::end(BenchFens)}; }

uint64_t run_bench(Searcher& s, int depth, bool print_each) {
    s.new_game();
    Limits lim;
    lim.depth = depth;
    s.stop = false;
    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();
    int n = 0;
    for (const char* fen : BenchFens) {
        Position pos;
        if (!pos.set_fen(fen)) { std::printf("bad bench fen: %s\n", fen); continue; }
        s.think(pos, {}, lim, false);
        total += s.nodes();
        ++n;
        if (print_each) std::printf("%2d: %llu  %s\n", n, (unsigned long long)s.nodes(), fen);
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("Nodes searched: %llu\nNodes/second: %.0f\n", (unsigned long long)total, total / std::max(sec, 1e-3));
    std::fflush(stdout);
    return total;
}
