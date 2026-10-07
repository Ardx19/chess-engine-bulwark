// Minimal test runner (no external deps). Usage: run_tests <data_dir> [--full]
#include "movegen.h"
#include "perft.h"
#include "bench.h"
#include "eval.h"
#include "search.h"
#include "tablebase.h"
#include <algorithm>
#include <cctype>
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

// Color flip: mirror ranks, swap piece colors, side to move, castling letters and ep rank.
static std::string color_flip(const std::string& fen) {
    std::istringstream is(fen);
    std::string pl, stm, cs, ep, rest;
    is >> pl >> stm >> cs >> ep;
    std::getline(is, rest);
    std::vector<std::string> rows;
    std::string cur;
    for (char c : pl) { if (c == '/') { rows.push_back(cur); cur.clear(); } else cur += c; }
    rows.push_back(cur);
    std::reverse(rows.begin(), rows.end());
    std::string out;
    for (size_t i = 0; i < rows.size(); ++i) {
        for (char c : rows[i]) out += std::isalpha((unsigned char)c) ? (std::isupper((unsigned char)c) ? (char)std::tolower(c) : (char)std::toupper(c)) : c;
        if (i + 1 < rows.size()) out += '/';
    }
    std::string c2;
    for (char c : cs) c2 += std::isalpha((unsigned char)c) ? (std::isupper((unsigned char)c) ? (char)std::tolower(c) : (char)std::toupper(c)) : c;
    if (ep != "-") ep[1] = char('1' + '8' - ep[1]);
    return out + (stm == "w" ? " b " : " w ") + c2 + " " + ep + rest;
}

static void test_eval_symmetry() {
    for (const char* f : bench_fens()) {
        Position a, b;
        a.set_fen(f); b.set_fen(color_flip(f));
        CHECK(Eval::evaluate(a) == Eval::evaluate(b), "eval asymmetry %s: %d vs %d", f, Eval::evaluate(a), Eval::evaluate(b));
        // flipped perft must agree too
        CHECK(perft(a, 3) == perft(b, 3), "flip perft %s", f);
    }
}

static void test_see() {
    struct T { const char* fen; const char* mv; int thr; bool exp; };
    T tests[] = {
        {"1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", "e1e5", 0, true},      // wins a pawn
        {"1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", 0, false},  // loses a knight for a pawn
        {"1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", -250, true},
    };
    for (auto& t : tests) {
        Position p; p.set_fen(t.fen);
        Move m = p.parse_move(t.mv);
        CHECK(m, "see move %s", t.mv);
        if (m) CHECK(p.see_ge(m, t.thr) == t.exp, "see %s %s thr %d", t.fen, t.mv, t.thr);
    }
}

static std::string uci_best(Searcher& s, const char* fen, int depth, int* score_out = nullptr) {
    Position p; p.set_fen(fen);
    Limits l; l.depth = depth;
    s.new_game();
    s.stop = false;
    Move m = s.think(p, {}, l, false);
    (void)score_out;
    return p.move_to_uci(m, false);
}

static void test_search() {
    Searcher s;
    CHECK(uci_best(s, "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 4) == "a1a8", "mate in 1");
    CHECK(uci_best(s, "k7/8/1K6/8/8/8/8/7R w - - 0 1", 4) == "h1h8", "mate in 1 (rook)");
    // stalemate trap: Qg6?? stalemates; engine with extra queen must not stalemate
    {
        Position p; p.set_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
        Limits l; l.depth = 6; s.new_game(); s.stop = false;
        Move m = s.think(p, {}, l, false);
        Position c = p.make(m);
        MoveList ml; generate_legal(c, ml);
        CHECK(ml.size() > 0 || c.in_check(), "engine stalemated the opponent: %s", p.move_to_uci(m, false).c_str());
    }
    // Draw by repetition inside history: K+R vs K+R shuffling must not crash and returns a legal move
    {
        Position p; p.set_fen("8/8/8/8/8/4k3/8/R3K3 w - - 0 1");
        Limits l; l.depth = 5; s.new_game(); s.stop = false;
        Move m = s.think(p, {}, l, false);
        CHECK(m && p.is_legal(m), "legal move in KR vs K");
    }
    // Nodes limit respected approximately
    {
        Position p; p.set_fen(Position::START_FEN);
        Limits l; l.nodes = 20000; s.new_game(); s.stop = false;
        s.think(p, {}, l, false);
        CHECK(s.nodes() < 40000, "node limit: %llu", (unsigned long long)s.nodes());
    }
}

// Needs SYZYGY_PATH (3-5 piece tables); skipped otherwise.
static void test_tablebase() {
    const char* path = std::getenv("SYZYGY_PATH");
    if (!path) { std::puts("tablebase test skipped (SYZYGY_PATH unset)"); return; }
    CHECK(TB::init(path) && TB::largest() >= 5, "tb init");
    Position p;
    p.set_fen("8/8/8/8/8/5k2/6p1/6K1 w - - 0 1");   // black pawn promotes: white loses
    CHECK(TB::probe_wdl(p) == TB::LOSS, "KvKP loss");
    p.set_fen("8/8/8/4k3/8/8/2K5/5Q2 w - - 0 1");   // KQK: won, root filter keeps only winning moves
    CHECK(TB::probe_wdl(p) == TB::WIN, "KQK win");
    MoveList allowed; CHECK(TB::root_filter(p, allowed) && allowed.size() > 0, "root filter KQK");
    p.set_fen("8/8/8/8/8/8/8/K1k5 w - - 0 1");      // bare kings: draw
    CHECK(TB::probe_wdl(p) == TB::DRAW || TB::probe_wdl(p) == TB::FAIL, "KK");
    // A stalemating Qg6?? type move must not be kept when the position is a win: 7k/5Q2/6K1 w
    p.set_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
    MoveList a2;
    if (TB::root_filter(p, a2))
        for (Move m : a2) { Position c = p.make(m); MoveList l; generate_legal(c, l); CHECK(l.size() > 0 || c.in_check(), "TB kept stalemating move"); }
    TB::init("");
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
    test_eval_symmetry();
    test_see();
    test_search();
    test_tablebase();
    run_perft_file(dir + "/perft_standard.epd", cap);
    run_perft_file(dir + "/perft_960.epd", cap);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
