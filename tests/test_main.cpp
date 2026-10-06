// Minimal test runner (no external deps). Usage: run_tests <data_dir> [--full]
#include "movegen.h"
#include "perft.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int failures = 0, checks = 0;
#define CHECK(cond, ...) do { ++checks; if (!(cond)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::puts(""); } } while (0)

// perft file format: "<fen> ;D1 n ;D2 n ..." ; lines starting with # are skipped.
static void run_perft_file(const std::string& path, U64 max_nodes) {
    std::ifstream in(path);
    CHECK(in.good(), "cannot open %s", path.c_str());
    std::string line;
    int positions = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t semi = line.find(';');
        std::string fen = line.substr(0, semi);
        Position pos;
        if (!pos.set_fen(fen)) { CHECK(false, "bad fen %s", fen.c_str()); continue; }
        std::istringstream rest(line.substr(semi));
        std::string tok;
        while (rest >> tok) {
            if (tok[0] != ';') { ; }
            int d = std::atoi(tok.c_str() + 2);
            U64 expect;
            if (!(rest >> expect)) break;
            if (expect > max_nodes) continue;
            U64 got = perft(pos, d);
            CHECK(got == expect, "perft(%d) %s: got %llu expected %llu", d, fen.c_str(), (unsigned long long)got, (unsigned long long)expect);
        }
        ++positions;
    }
    std::printf("%s: %d positions\n", path.c_str(), positions);
}

static void test_fen_roundtrip() {
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"};
    for (auto f : fens) {
        Position p;
        CHECK(p.set_fen(f), "parse %s", f);
        CHECK(p.fen() == f, "roundtrip %s -> %s", f, p.fen().c_str());
    }
    // Shredder-FEN and X-FEN agree
    Position a, b;
    a.set_fen("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9");
    b.set_fen("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w KQkq - 2 9");
    CHECK(a.key == b.key && a.castling == 15, "960 castling parse");
    CHECK(a.fen(true) == "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9", "shredder out: %s", a.fen(true).c_str());
}

// Incremental key must equal the recomputed key along every perft path to depth 3.
static void zobrist_walk(const Position& p, int d) {
    CHECK(p.key == p.compute_key(), "zobrist mismatch %s", p.fen().c_str());
    if (!d) return;
    MoveList ml; generate_legal(p, ml);
    for (Move m : ml) zobrist_walk(p.make(m), d - 1);
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "tests/data";
    bool full = argc > 2 && std::string(argv[2]) == "--full";
    Bitboards::init();
    Zobrist::init();
    test_fen_roundtrip();
    for (const char* f : {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                          "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9"}) {
        Position p; p.set_fen(f); zobrist_walk(p, 3);
    }
    U64 cap = full ? ~0ULL : 20000000ULL;
    run_perft_file(dir + "/perft_standard.epd", cap);
    run_perft_file(dir + "/perft_960.epd", cap);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
